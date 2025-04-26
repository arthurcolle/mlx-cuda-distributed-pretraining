from mlx_optimizers.shampoo import Shampoo, ShampooParams
from mlx_optimizers.muon import Muon as MuonOptimizer

# Rename to avoid circular import
Muon = MuonOptimizer
from mlx_optimizers.hybrid_optimizer import HybridOptimizer

__all__ = [
    "Shampoo",
    "ShampooParams",
    "Muon",
    "HybridOptimizer",
]