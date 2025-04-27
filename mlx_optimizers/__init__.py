import logging

logger = logging.getLogger(__name__)

try:
    from mlx_optimizers.shampoo import Shampoo, ShampooParams
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
    class ShampooParams: pass
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
