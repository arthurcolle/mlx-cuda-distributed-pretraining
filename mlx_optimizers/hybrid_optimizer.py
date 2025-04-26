"""
Hybrid optimizer implementation for MLX
Combines multiple optimizers for different parameter types
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten
from typing import Any, Callable, Dict, Optional, Tuple, Union, List

from mlx_optimizers.shampoo import Shampoo, ShampooParams
from mlx_optimizers.muon import Muon


class HybridOptimizer(optim.Optimizer):
    """
    Hybrid optimizer that applies different optimization strategies to different parameter types
    
    This optimizer allows using specialized optimizers for different parameter shapes:
    - Matrix parameters (2D): Often benefit from geometric optimizers like Muon
    - Other parameters: Can use different approaches like Shampoo or AdamW
    
    Args:
        learning_rate: Learning rate or learning rate schedule
        matrix_optimizer: Optimizer to use for 2D matrix parameters
        non_matrix_optimizer: Optimizer to use for other parameter shapes
        parameter_mapping: Optional dict mapping parameter names to specific optimizers
    """
    
    def __init__(
        self,
        learning_rate: Union[float, Callable] = 0.01,
        matrix_optimizer: Optional[optim.Optimizer] = None,
        non_matrix_optimizer: Optional[optim.Optimizer] = None,
        parameter_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        
        # Store learning rate
        self._learning_rate = learning_rate
        
        # Initialize optimizers
        self.matrix_optimizer = matrix_optimizer
        self.non_matrix_optimizer = non_matrix_optimizer
        
        # Parameter mapping for custom optimizer assignment
        self.parameter_mapping = parameter_mapping or {}
        
        # Initialize state
        self.state = {}
        self.count = 0
        
        # Pass count updates to sub-optimizers
        if hasattr(self.matrix_optimizer, 'count'):
            self.matrix_optimizer.count = self.count
        if hasattr(self.non_matrix_optimizer, 'count'):
            self.non_matrix_optimizer.count = self.count

    def update(self, model: nn.Module, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Update parameters using appropriate optimizers based on parameter shape
        
        Args:
            model: The model to update
            gradients: Gradients for each parameter
            
        Returns:
            Dictionary of parameter updates
        """
        updates = {}
        matrix_params = {}
        matrix_grads = {}
        non_matrix_params = {}
        non_matrix_grads = {}
        
        # Sort parameters by shape
        for name, param in tree_flatten(model.parameters()):
            grad = gradients.get(name)
            if grad is None:
                continue
                
            # Check if parameter has a specific optimizer assigned
            if name in self.parameter_mapping:
                optimizer_type = self.parameter_mapping[name]
                if optimizer_type == "matrix":
                    matrix_params[name] = param
                    matrix_grads[name] = grad
                else:
                    non_matrix_params[name] = param
                    non_matrix_grads[name] = grad
            # Otherwise use shape-based assignment
            elif len(param.shape) == 2:
                # 2D parameters (matrices) go to matrix optimizer
                matrix_params[name] = param
                matrix_grads[name] = grad
            else:
                # Other shapes go to non-matrix optimizer
                non_matrix_params[name] = param
                non_matrix_grads[name] = grad
        
        # Update with matrix optimizer if available and we have matrix params
        if self.matrix_optimizer is not None and matrix_params:
            matrix_updates = self.matrix_optimizer.update(
                nn.Module(matrix_params), matrix_grads
            )
            updates.update(matrix_updates)
        
        # Update with non-matrix optimizer if available and we have non-matrix params
        if self.non_matrix_optimizer is not None and non_matrix_params:
            non_matrix_updates = self.non_matrix_optimizer.update(
                nn.Module(non_matrix_params), non_matrix_grads
            )
            updates.update(non_matrix_updates)
            
        # Count updates
        self.count += 1
        
        # Sync counts with sub-optimizers
        if hasattr(self.matrix_optimizer, 'count'):
            self.matrix_optimizer.count = self.count
        if hasattr(self.non_matrix_optimizer, 'count'):
            self.non_matrix_optimizer.count = self.count
            
        return updates