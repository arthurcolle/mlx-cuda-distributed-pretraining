import logging

logger = logging.getLogger(__name__)

# Define ShampooParams here to ensure it's always available
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

try:
    from dataclasses import dataclass
    from mlx_optimizers.shampoo import Shampoo
    from mlx_optimizers.muon import Muon
    from mlx_optimizers.enhanced_optimizers import AdamWEnhanced, SGDEnhanced, LionEnhanced
    
    try:
        from mlx_optimizers.hybrid_optimizer import HybridOptimizer
    except ImportError:
        logger.warning("HybridOptimizer could not be imported")
        HybridOptimizer = None

except ImportError as e:
    logger.error(f"Error importing MLX optimizers: {e}")
    # Provide fallback empty classes to prevent import errors
    class Shampoo: pass
    class Muon: pass
    class AdamWEnhanced: pass
    class SGDEnhanced: pass
    class LionEnhanced: pass
    HybridOptimizer = None

__all__ = [
    "Shampoo",
    "ShampooParams",
    "Muon",
    "HybridOptimizer",
    "AdamWEnhanced",
    "SGDEnhanced",
    "LionEnhanced",
]
