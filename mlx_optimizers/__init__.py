from mlx_optimizers.shampoo import Shampoo, ShampooParams
from mlx_optimizers.muon import Muon
from mlx_optimizers.enhanced_optimizers import AdamWEnhanced, SGDEnhanced, LionEnhanced

try:
    from mlx_optimizers.hybrid_optimizer import HybridOptimizer
except ImportError:
    # Create a placeholder if the file doesn't exist yet
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
