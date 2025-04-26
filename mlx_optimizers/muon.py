"""Muon optimizer implementation for MLX."""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any


@dataclass
class MuonState:
    """State for the Muon optimizer."""
    count: mx.array
    momentum: Dict[str, mx.array]
    velocity: Dict[str, mx.array]


class Muon(nn.Module):
    """Muon optimizer implementation.
    
    A variant of Adam with improved convergence properties.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
    def zero_grad(self) -> None:
        """Reset the gradients."""
        pass
        
    def init(self, model: nn.Module) -> MuonState:
        """Initialize the optimizer state for the model parameters."""
        params = dict(model.parameters())
        momentum = {k: mx.zeros_like(p) for k, p in params.items()}
        velocity = {k: mx.zeros_like(p) for k, p in params.items()}
        return MuonState(count=mx.array(0), momentum=momentum, velocity=velocity)
        
    def update(
        self, model: nn.Module, gradients: Dict[str, mx.array]
    ) -> nn.Module:
        """Update the model parameters using the Muon optimizer."""
        # Initialize state if not already set
        if not hasattr(self, 'state'):
            self.state = self.init(model)
            
        # In case of error, re-initialize state to make sure it has the right structure
        if not isinstance(self.state, MuonState):
            self.state = self.init(model)
        
        # Update the state
        model, self.state = self._update_with_state(model, gradients, self.state)
        return model
        
    def _update_with_state(
        self, model: nn.Module, gradients: Dict[str, mx.array], state: MuonState
    ) -> Tuple[nn.Module, MuonState]:
        """Internal method to update the model parameters and return the updated state."""
            
        params = dict(model.parameters())
        count = state.count + 1
        momentum = state.momentum
        velocity = state.velocity
        lr = self.learning_rate
        beta1, beta2 = self.beta1, self.beta2
        eps = self.eps
        weight_decay = self.weight_decay
        
        new_params = {}
        new_momentum = {}
        new_velocity = {}
        
        # Initialize the momentum and velocity for any new parameters
        for k, p in params.items():
            if k not in momentum:
                momentum[k] = mx.zeros_like(p)
            if k not in velocity:
                velocity[k] = mx.zeros_like(p)
                
            # If the parameter doesn't have a gradient, keep it as is
            if k not in gradients:
                new_params[k] = p
                new_momentum[k] = momentum[k]
                new_velocity[k] = velocity[k]
                continue
                
            g = gradients[k]
            
            if weight_decay != 0.0:
                g = g + weight_decay * p
                
            m = beta1 * momentum[k] + (1.0 - beta1) * g
            v = beta2 * velocity[k] + (1.0 - beta2) * (g * g)
            
            # Bias correction
            m_hat = m / (1.0 - beta1 ** count)
            v_hat = v / (1.0 - beta2 ** count)
            
            # Muon adjustment - smoother step size calculation with better numerical properties
            step = lr * m_hat / (mx.sqrt(v_hat) + eps)
            
            new_params[k] = p - step
            new_momentum[k] = m
            new_velocity[k] = v
            
        new_model = model.replace_params(new_params)
        new_state = MuonState(count=count, momentum=new_momentum, velocity=new_velocity)
        
        return new_model, new_state