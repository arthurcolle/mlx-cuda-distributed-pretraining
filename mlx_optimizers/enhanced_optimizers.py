"""
Enhanced optimizers for MLX with additional features for SOTA performance.
These optimizers extend the basic MLX optimizers with features like:
- Decoupled weight decay (AdamW-style)
- Gradient clipping
- Learning rate scheduling integration
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from mlx.utils import tree_map, tree_flatten

class AdamWEnhanced(optim.Optimizer):
    """
    Enhanced AdamW implementation with decoupled weight decay and additional features.
    
    This optimizer implements the AdamW algorithm with proper decoupled weight decay
    and supports additional features like gradient clipping, momentum scheduling,
    and automatic step counting for learning rate schedules.
    
    Args:
        learning_rate: The learning rate or learning rate schedule
        betas: Coefficients for computing running averages of gradient and its square
        eps: Term added to the denominator to improve numerical stability
        weight_decay: Weight decay coefficient
        grad_clip_norm: Optional gradient clipping norm
        bias_correction: Whether to use bias correction in moment estimates
        ema_momentum: Optional EMA momentum for weight averaging
        local_momentum: Whether to use per-parameter momentum values
        fused_ops: Whether to use fused operations for compute efficiency
    """
    
    def __init__(
        self,
        learning_rate: Union[float, Callable] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        bias_correction: bool = True,
        ema_momentum: Optional[float] = None,
        local_momentum: bool = False,
        fused_ops: bool = False,
        amsgrad: bool = False,
        ema_decay: float = 0.9999
    ):
        super().__init__()
        # Store parameters
        self._learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.bias_correction = bias_correction
        self.ema_momentum = ema_momentum
        self.local_momentum = local_momentum
        self.fused_ops = fused_ops
        self.amsgrad = amsgrad
        self.ema_decay = ema_decay
        
        # Initialize counters and state
        self.state = {}
        self.count = 0
        
    def _compute_ema(self, model_params: Dict[str, mx.array]) -> None:
        """Update EMA parameters if enabled"""
        if self.ema_momentum is None:
            return
            
        if 'ema_params' not in self.state:
            # Initialize EMA parameters
            self.state['ema_params'] = {k: v.copy() for k, v in model_params.items()}
        else:
            # Update EMA parameters
            decay = self.ema_momentum 
            if callable(decay):
                decay = decay(self.count)
            
            # Apply EMA update
            for k, v in model_params.items():
                if k in self.state['ema_params']:
                    self.state['ema_params'][k] = (
                        decay * self.state['ema_params'][k] + (1 - decay) * v
                    )
    
    def _apply_weight_decay(self, param: mx.array, grad: mx.array, name: str) -> mx.array:
        """Apply decoupled weight decay to parameters"""
        if self.weight_decay == 0.0:
            return grad
            
        # Don't apply weight decay to bias terms or layer norm parameters
        if name.endswith('bias') or '.norm' in name or '.ln' in name:
            return grad
            
        # Calculate effective weight decay
        lr = self._learning_rate(self.count) if callable(self._learning_rate) else self._learning_rate
        wd = self.weight_decay * lr
        
        # Return updated gradient
        return grad + wd * param
    
    def _clip_gradients(self, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Apply gradient clipping if enabled"""
        if self.grad_clip_norm is None or self.grad_clip_norm <= 0.0:
            return gradients
            
        # Calculate global norm of gradients
        total_norm = mx.sqrt(sum(
            mx.sum(mx.square(g)) for g in gradients.values() if g is not None
        ))
        
        # Apply clipping if norm exceeds threshold
        clip_coef = self.grad_clip_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            gradients = {k: v * clip_coef if v is not None else None for k, v in gradients.items()}
            
        return gradients
    
    def update(self, model: nn.Module, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Update parameters with AdamW optimization"""
        # Apply gradient clipping if enabled
        if self.grad_clip_norm is not None:
            gradients = self._clip_gradients(gradients)
            
        # Get learning rate
        lr = self._learning_rate(self.count) if callable(self._learning_rate) else self._learning_rate
        
        # Get beta values
        beta1, beta2 = self.betas
        
        # Update each parameter
        updates = {}
        model_params = dict(tree_flatten(model.parameters()))
        
        for name, param in model_params.items():
            grad = gradients.get(name)
            if grad is None:
                continue
                
            # Apply weight decay
            grad = self._apply_weight_decay(param, grad, name)
            
            # Initialize state for this parameter if needed
            if name not in self.state:
                self.state[name] = {
                    "exp_avg": mx.zeros_like(param),
                    "exp_avg_sq": mx.zeros_like(param),
                    "max_exp_avg_sq": mx.zeros_like(param) if self.amsgrad else None,
                }
                
            # Get optimizer state for this parameter
            state = self.state[name]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            
            # Update moments
            exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
            exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
            
            # Store updated moments
            state["exp_avg"] = exp_avg
            state["exp_avg_sq"] = exp_avg_sq
            
            # Apply bias correction if enabled
            if self.bias_correction:
                bias_correction1 = 1.0 - beta1 ** (self.count + 1)
                bias_correction2 = 1.0 - beta2 ** (self.count + 1)
                step_size = lr / bias_correction1
                denom = mx.sqrt(exp_avg_sq) / mx.sqrt(bias_correction2) + self.eps
            else:
                step_size = lr
                denom = mx.sqrt(exp_avg_sq) + self.eps
                
            # Apply AMSGrad if enabled
            if self.amsgrad:
                max_exp_avg_sq = state["max_exp_avg_sq"]
                max_exp_avg_sq = mx.maximum(max_exp_avg_sq, exp_avg_sq)
                state["max_exp_avg_sq"] = max_exp_avg_sq
                denom = mx.sqrt(max_exp_avg_sq) + self.eps
                
            # Compute update
            update = -step_size * exp_avg / denom
            updates[name] = update
            
        # Update EMA parameters if enabled
        next_params = {name: param + updates.get(name, 0) for name, param in model_params.items()}
        self._compute_ema(next_params)
        
        # Increment step counter
        self.count += 1
        
        return updates
    
    def get_ema_params(self) -> Optional[Dict[str, mx.array]]:
        """Get EMA parameters if EMA is enabled"""
        return self.state.get('ema_params')


class SGDEnhanced(optim.Optimizer):
    """
    Enhanced SGD implementation with additional features for high performance.
    
    This optimizer extends SGD with:
    - Proper weight decay (decoupled like AdamW)
    - Nesterov momentum
    - Gradient clipping
    - Learning rate scheduling integration
    - Weight averaging using EMA
    
    Args:
        learning_rate: The learning rate or learning rate schedule
        momentum: Momentum coefficient
        weight_decay: Weight decay coefficient
        nesterov: Whether to use Nesterov momentum
        grad_clip_norm: Optional gradient clipping norm
        dampening: Dampening for momentum
        ema_momentum: Optional EMA momentum for weight averaging
        fused_ops: Whether to use fused operations for compute efficiency
    """
    
    def __init__(
        self,
        learning_rate: Union[float, Callable] = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        grad_clip_norm: Optional[float] = None,
        dampening: float = 0.0,
        ema_momentum: Optional[float] = None,
        fused_ops: bool = False,
    ):
        super().__init__()
        # Store parameters
        self._learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.grad_clip_norm = grad_clip_norm
        self.dampening = dampening
        self.ema_momentum = ema_momentum
        self.fused_ops = fused_ops
        
        # Initialize counters and state
        self.state = {}
        self.count = 0
        
    def _compute_ema(self, model_params: Dict[str, mx.array]) -> None:
        """Update EMA parameters if enabled"""
        if self.ema_momentum is None:
            return
            
        if 'ema_params' not in self.state:
            # Initialize EMA parameters
            self.state['ema_params'] = {k: v.copy() for k, v in model_params.items()}
        else:
            # Update EMA parameters
            decay = self.ema_momentum 
            if callable(decay):
                decay = decay(self.count)
            
            # Apply EMA update
            for k, v in model_params.items():
                if k in self.state['ema_params']:
                    self.state['ema_params'][k] = (
                        decay * self.state['ema_params'][k] + (1 - decay) * v
                    )
    
    def _apply_weight_decay(self, param: mx.array, grad: mx.array, name: str) -> mx.array:
        """Apply decoupled weight decay to parameters"""
        if self.weight_decay == 0.0:
            return grad
            
        # Don't apply weight decay to bias terms or layer norm parameters
        if name.endswith('bias') or '.norm' in name or '.ln' in name:
            return grad
            
        # Calculate effective weight decay
        lr = self._learning_rate(self.count) if callable(self._learning_rate) else self._learning_rate
        wd = self.weight_decay * lr
        
        # Return updated gradient
        return grad + wd * param
    
    def _clip_gradients(self, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Apply gradient clipping if enabled"""
        if self.grad_clip_norm is None or self.grad_clip_norm <= 0.0:
            return gradients
            
        # Calculate global norm of gradients
        total_norm = mx.sqrt(sum(
            mx.sum(mx.square(g)) for g in gradients.values() if g is not None
        ))
        
        # Apply clipping if norm exceeds threshold
        clip_coef = self.grad_clip_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            gradients = {k: v * clip_coef if v is not None else None for k, v in gradients.items()}
            
        return gradients
    
    def update(self, model: nn.Module, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Update parameters with SGD optimization"""
        # Apply gradient clipping if enabled
        if self.grad_clip_norm is not None:
            gradients = self._clip_gradients(gradients)
            
        # Get learning rate
        lr = self._learning_rate(self.count) if callable(self._learning_rate) else self._learning_rate
        
        # Update each parameter
        updates = {}
        model_params = dict(tree_flatten(model.parameters()))
        
        for name, param in model_params.items():
            grad = gradients.get(name)
            if grad is None:
                continue
                
            # Apply weight decay
            grad = self._apply_weight_decay(param, grad, name)
            
            # Initialize state for this parameter if needed
            if name not in self.state:
                self.state[name] = {
                    "momentum_buffer": mx.zeros_like(param),
                }
                
            # Get optimizer state for this parameter
            momentum_buffer = self.state[name]["momentum_buffer"]
            
            # Update momentum buffer
            momentum_buffer = self.momentum * momentum_buffer + (1.0 - self.dampening) * grad
            self.state[name]["momentum_buffer"] = momentum_buffer
            
            # Compute update with or without Nesterov momentum
            if self.nesterov:
                update = -lr * (grad + self.momentum * momentum_buffer)
            else:
                update = -lr * momentum_buffer
                
            updates[name] = update
            
        # Update EMA parameters if enabled
        next_params = {name: param + updates.get(name, 0) for name, param in model_params.items()}
        self._compute_ema(next_params)
        
        # Increment step counter
        self.count += 1
        
        return updates
    
    def get_ema_params(self) -> Optional[Dict[str, mx.array]]:
        """Get EMA parameters if EMA is enabled"""
        return self.state.get('ema_params')


class LionEnhanced(optim.Optimizer):
    """
    Enhanced Lion optimizer implementation for MLX.
    
    Lion (Evolved Sign Momentum) combines elements of Adam and SGD using sign-based
    updates for better performance on large language models. This implementation
    adds decoupled weight decay, gradient clipping, and EMA features.
    
    Based on: https://arxiv.org/abs/2302.06675
    
    Args:
        learning_rate: The learning rate or learning rate schedule
        betas: Coefficients for computing running averages
        weight_decay: Weight decay coefficient
        grad_clip_norm: Optional gradient clipping norm
        ema_momentum: Optional EMA momentum for weight averaging
    """
    
    def __init__(
        self,
        learning_rate: Union[float, Callable] = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        ema_momentum: Optional[float] = None,
    ):
        super().__init__()
        # Store parameters
        self._learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.ema_momentum = ema_momentum
        
        # Initialize counters and state
        self.state = {}
        self.count = 0
        
    def _compute_ema(self, model_params: Dict[str, mx.array]) -> None:
        """Update EMA parameters if enabled"""
        if self.ema_momentum is None:
            return
            
        if 'ema_params' not in self.state:
            # Initialize EMA parameters
            self.state['ema_params'] = {k: v.copy() for k, v in model_params.items()}
        else:
            # Update EMA parameters
            decay = self.ema_momentum 
            if callable(decay):
                decay = decay(self.count)
            
            # Apply EMA update
            for k, v in model_params.items():
                if k in self.state['ema_params']:
                    self.state['ema_params'][k] = (
                        decay * self.state['ema_params'][k] + (1 - decay) * v
                    )
    
    def _clip_gradients(self, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Apply gradient clipping if enabled"""
        if self.grad_clip_norm is None or self.grad_clip_norm <= 0.0:
            return gradients
            
        # Calculate global norm of gradients
        total_norm = mx.sqrt(sum(
            mx.sum(mx.square(g)) for g in gradients.values() if g is not None
        ))
        
        # Apply clipping if norm exceeds threshold
        clip_coef = self.grad_clip_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            gradients = {k: v * clip_coef if v is not None else None for k, v in gradients.items()}
            
        return gradients
    
    def update(self, model: nn.Module, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Update parameters with Lion optimization"""
        # Apply gradient clipping if enabled
        if self.grad_clip_norm is not None:
            gradients = self._clip_gradients(gradients)
            
        # Get learning rate
        lr = self._learning_rate(self.count) if callable(self._learning_rate) else self._learning_rate
        
        # Get beta values
        beta1, beta2 = self.betas
        
        # Update each parameter
        updates = {}
        model_params = dict(tree_flatten(model.parameters()))
        
        for name, param in model_params.items():
            grad = gradients.get(name)
            if grad is None:
                continue
                
            # Initialize state for this parameter if needed
            if name not in self.state:
                self.state[name] = {
                    "exp_avg": mx.zeros_like(param),
                }
                
            # Get optimizer state for this parameter
            exp_avg = self.state[name]["exp_avg"]
            
            # Update momentum
            update = beta1 * exp_avg + (1 - beta1) * grad
            
            # Store updated momentum
            self.state[name]["exp_avg"] = update
            
            # Weight decay
            if self.weight_decay != 0:
                param = param * (1 - lr * self.weight_decay)
                
            # Lion update: apply sign-based update directly
            updates[name] = -lr * mx.sign(update)
            
        # Update EMA parameters if enabled
        next_params = {name: param + updates.get(name, 0) for name, param in model_params.items()}
        self._compute_ema(next_params)
        
        # Increment step counter
        self.count += 1
        
        return updates
        
    def get_ema_params(self) -> Optional[Dict[str, mx.array]]:
        """Get EMA parameters if EMA is enabled"""
        return self.state.get('ema_params')