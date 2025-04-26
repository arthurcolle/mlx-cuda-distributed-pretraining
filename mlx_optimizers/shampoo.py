"""
Distributed Shampoo optimizer for MLX
Based on: https://arxiv.org/abs/1802.09568

This implementation provides a memory-efficient and computationally efficient 
version of the Shampoo optimizer, which approximates second-order optimization
using Kronecker factorization.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten


@dataclass
class ShampooParams:
    """Parameters for the Shampoo optimizer"""
    beta1: float = 0.9
    beta2: float = 0.99
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    update_period: int = 1  # Update preconditioners every N steps
    start_preconditioning_step: int = 10  # Start using preconditioners after N steps
    preconditioner_epsilon: float = 1e-6  # Regularization for preconditioners
    max_preconditioner_dim: int = 1024  # Maximum dimension for preconditioners
    exponent_override: float = 0.75  # Root exponent value for preconditioners
    use_bias_correction: bool = True  # Whether to use bias correction
    grafting_optimizer: str = "adam"  # Optimizer to use for grafting
    use_decoupled_weight_decay: bool = True  # Whether to use decoupled weight decay

    def __post_init__(self):
        assert 0.0 <= self.beta1 < 1.0, "beta1 must be in [0, 1)"
        assert 0.0 <= self.beta2 < 1.0, "beta2 must be in [0, 1)"
        assert self.epsilon > 0.0, "epsilon must be positive"
        assert self.update_period > 0, "update_period must be positive"
        assert self.start_preconditioning_step >= 0, "start_preconditioning_step must be non-negative"
        assert self.max_preconditioner_dim > 0, "max_preconditioner_dim must be positive"
        assert 0.0 < self.exponent_override <= 1.0, "exponent_override must be in (0, 1]"
        assert self.grafting_optimizer in ["sgd", "adam", "momentum"], \
            "grafting_optimizer must be one of 'sgd', 'adam', 'momentum'"


class MatrixSqrt:
    """Utility class for computing the matrix square root using Newton iterations"""
    
    @staticmethod
    def symmetric_matrix_sqrt(matrix: mx.array, epsilon: float = 1e-6, num_iters: int = 6) -> mx.array:
        """
        Compute the square root of a symmetric positive definite matrix
        using Newton's method.
        
        Args:
            matrix: A symmetric positive definite matrix
            epsilon: Small constant for numerical stability
            num_iters: Number of iterations for Newton's method
            
        Returns:
            Matrix square root of the input matrix
        """
        # Add small epsilon to diagonal for numerical stability
        matrix = matrix + mx.eye(matrix.shape[0]) * epsilon
        
        # Initialize Y as the identity matrix
        Y = mx.eye(matrix.shape[0])
        
        # Initialize Z as the input matrix
        Z = matrix
        
        # Iterative improvement using Newton's method
        for _ in range(num_iters):
            # T = 0.5 * (3 * I - Z^T * Y)
            T = 0.5 * (3.0 * mx.eye(matrix.shape[0]) - mx.matmul(Z, Y))
            
            # Y = Y * T
            Y = mx.matmul(Y, T)
            
            # Z = T * Z
            Z = mx.matmul(T, Z)
        
        return Y
    
    @staticmethod
    def matrix_inverse_pth_root(matrix: mx.array, p: float, epsilon: float = 1e-6, num_iters: int = 6) -> mx.array:
        """
        Compute the inverse pth root of a symmetric positive definite matrix
        using Newton's method.
        
        Args:
            matrix: A symmetric positive definite matrix
            p: The exponent (typically 0.5 or 0.75)
            epsilon: Small constant for numerical stability
            num_iters: Number of iterations for Newton's method
            
        Returns:
            Inverse pth root of the input matrix
        """
        # Add small epsilon to diagonal for numerical stability
        matrix = matrix + mx.eye(matrix.shape[0]) * epsilon
        
        # Calculate constants for the iterations
        alpha = -1.0 / p
        beta = 1.0
        
        # Initialize Z as the normalized matrix
        Z = matrix / mx.trace(matrix)
        
        # Initialize scaling factor
        scaling_factor = mx.trace(matrix) ** (1.0 / p)
        
        # Iterative improvement
        for _ in range(num_iters):
            # Calculate M = (I/beta - Z/alpha)
            M = mx.eye(Z.shape[0]) * beta - Z * alpha
            
            # Update Z = Z * M
            Z = mx.matmul(Z, M)
        
        # Apply scaling
        Z = Z * (scaling_factor ** alpha)
        
        return Z


class Shampoo(optim.Optimizer):
    """
    Distributed Shampoo optimizer implementation for MLX
    
    This optimizer implements the Shampoo algorithm, which uses matrix factorization 
    to approximate second-order optimization. It maintains preconditioners for each 
    parameter matrix.
    
    Args:
        learning_rate: Learning rate or learning rate schedule
        params: Shampoo parameters configuration
        use_distributed: Whether to use distributed computation
    """
    
    def __init__(
        self,
        learning_rate: Union[float, Callable] = 0.01,
        params: Optional[ShampooParams] = None,
        use_distributed: bool = False,
    ):
        super().__init__()
        
        # Initialize parameters
        self.params = params or ShampooParams()
        self._learning_rate = learning_rate
        self.use_distributed = use_distributed
        
        # Initialize state
        self.state = {}
        self.count = 0
        
        # Initialize grafting optimizer based on configuration
        if self.params.grafting_optimizer == "adam":
            self.grafting_optimizer = optim.Adam(
                learning_rate=learning_rate,
                betas=(self.params.beta1, self.params.beta2),
                eps=self.params.epsilon,
                # MLX's Adam doesn't support weight_decay, handle separately
            )
        elif self.params.grafting_optimizer == "momentum":
            self.grafting_optimizer = optim.SGD(
                learning_rate=learning_rate,
                momentum=self.params.beta1,
                # MLX's SGD doesn't support weight_decay, handle separately
            )
        else:  # sgd
            self.grafting_optimizer = optim.SGD(
                learning_rate=learning_rate,
                # MLX's SGD doesn't support weight_decay, handle separately
            )

    def _init_state(self, param: mx.array, name: str) -> Dict:
        """Initialize optimization state for a parameter"""
        shape = param.shape
        dtype = param.dtype
        
        # Create state dictionary
        param_state = {
            "momentum": mx.zeros_like(param),
            "preconditioners": None,
            "adagrad": mx.zeros_like(param),
            "statistics": None,
        }
        
        # Only create preconditioners for 2D parameters
        if len(shape) == 2:
            dim1, dim2 = shape
            
            # Limit dimensions for efficiency
            dim1 = min(dim1, self.params.max_preconditioner_dim)
            dim2 = min(dim2, self.params.max_preconditioner_dim)
            
            # Initialize statistics matrices
            statistics1 = mx.zeros((dim1, dim1), dtype=mx.float32)
            statistics2 = mx.zeros((dim2, dim2), dtype=mx.float32)
            
            param_state["statistics"] = [statistics1, statistics2]
            param_state["preconditioners"] = [None, None]
        
        return param_state

    def _compute_preconditioners(self, state: Dict, step: int) -> None:
        """Compute preconditioner matrices from statistics"""
        if state["statistics"] is None or step < self.params.start_preconditioning_step:
            return
        
        # Only compute preconditioners periodically
        if step % self.params.update_period != 0:
            return
        
        # Compute preconditioners for each dimension using matrix inverse pth root
        state["preconditioners"] = [
            MatrixSqrt.matrix_inverse_pth_root(
                stat, 
                p=self.params.exponent_override,
                epsilon=self.params.preconditioner_epsilon
            ) if stat is not None else None
            for stat in state["statistics"]
        ]

    def _update_statistics(self, state: Dict, grad: mx.array) -> None:
        """Update statistics for preconditioner computation"""
        if state["statistics"] is None:
            return
        
        # Extract necessary components
        statistics = state["statistics"]
        shape = grad.shape
        
        # If gradient is 2D, update statistics for both dimensions
        if len(shape) == 2:
            m, n = shape
            
            # Limit dimensions for efficiency
            m = min(m, self.params.max_preconditioner_dim)
            n = min(n, self.params.max_preconditioner_dim)
            
            # Extract limited gradient for efficiency
            limited_grad = grad[:m, :n]
            
            # Update Row statistics: G * G^T
            row_grad = mx.matmul(limited_grad, limited_grad.T)
            statistics[0] = self.params.beta2 * statistics[0] + (1 - self.params.beta2) * row_grad
            
            # Update Column statistics: G^T * G
            col_grad = mx.matmul(limited_grad.T, limited_grad)
            statistics[1] = self.params.beta2 * statistics[1] + (1 - self.params.beta2) * col_grad

    def _apply_preconditioners(self, state: Dict, grad: mx.array, step: int) -> mx.array:
        """Apply preconditioners to gradient"""
        # Initialize a copy of the gradient
        preconditioned_grad = grad.copy()
        
        # Only apply preconditioners if they exist and we're past the start step
        if (
            state["statistics"] is not None 
            and state["preconditioners"] is not None
            and step >= self.params.start_preconditioning_step
        ):
            shape = grad.shape
            
            # Only apply to 2D gradients
            if len(shape) == 2:
                m, n = shape
                
                # Limit dimensions for efficiency
                m_limited = min(m, self.params.max_preconditioner_dim)
                n_limited = min(n, self.params.max_preconditioner_dim)
                
                # Extract the part of the gradient to precondition
                limited_grad = grad[:m_limited, :n_limited]
                
                # Get preconditioners
                left_precond = state["preconditioners"][0]
                right_precond = state["preconditioners"][1]
                
                # Apply left and right preconditioners: L * G * R
                if left_precond is not None and right_precond is not None:
                    preconditioned_limited = mx.matmul(
                        mx.matmul(left_precond, limited_grad),
                        right_precond
                    )
                    
                    # Update the limited part of the gradient
                    preconditioned_grad = preconditioned_grad.at[:m_limited, :n_limited].set(preconditioned_limited)
        
        return preconditioned_grad

    def _apply_grafting(self, grafting_update: mx.array, shampoo_update: mx.array) -> mx.array:
        """Apply grafting by maintaining the direction of Shampoo but the magnitude of the grafting optimizer"""
        # Calculate norms
        shampoo_norm = mx.linalg.norm(shampoo_update)
        grafting_norm = mx.linalg.norm(grafting_update)
        
        # If either norm is zero, return the other update
        if shampoo_norm == 0:
            return grafting_update
        if grafting_norm == 0:
            return shampoo_update
        
        # Scale Shampoo update to have the same norm as the grafting update
        scaled_shampoo = shampoo_update * (grafting_norm / shampoo_norm)
        
        return scaled_shampoo

    def update(self, model: nn.Module, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Update parameters using the Shampoo optimizer"""
        updates = {}
        grafting_updates = {}
        self.count += 1
        
        # Get the current learning rate
        lr = self._learning_rate(self.count) if callable(self._learning_rate) else self._learning_rate
        
        # Update the grafting optimizer first
        if self.params.grafting_optimizer != "none":
            # If grafting uses its own optimizer, update it
            grafting_updates = self.grafting_optimizer.update(model, gradients)
        
        # Process each parameter
        for name, param in tree_flatten(model.parameters()):
            grad = gradients.get(name)
            if grad is None:
                continue
            
            # Initialize state if needed
            if name not in self.state:
                self.state[name] = self._init_state(param, name)
            
            state = self.state[name]
            
            # Apply weight decay to gradient if not using decoupled weight decay
            if self.params.weight_decay > 0.0 and not self.params.use_decoupled_weight_decay:
                grad = grad + self.params.weight_decay * param
            
            # Update statistics for preconditioners
            self._update_statistics(state, grad)
            
            # Compute preconditioners if needed
            self._compute_preconditioners(state, self.count)
            
            # Update momentum
            state["momentum"] = (
                self.params.beta1 * state["momentum"] + (1 - self.params.beta1) * grad
            )
            
            # Apply bias correction to momentum if enabled
            momentum = state["momentum"]
            if self.params.use_bias_correction:
                bias_correction = 1.0 - self.params.beta1 ** self.count
                momentum = momentum / bias_correction
            
            # Apply preconditioners to momentum
            preconditioned_grad = self._apply_preconditioners(state, momentum, self.count)
            
            # Compute update
            update = -lr * preconditioned_grad
            
            # Apply grafting if enabled
            if name in grafting_updates:
                update = self._apply_grafting(grafting_updates[name], update)
            
            # Apply decoupled weight decay if enabled
            if self.params.weight_decay > 0.0 and self.params.use_decoupled_weight_decay:
                update = update - lr * self.params.weight_decay * param
            
            # Store update
            updates[name] = update
        
        return updates