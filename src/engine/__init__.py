"""
Engine package initializer.

This module re-exports the most commonly used engine components so callers
can import them from `src.engine` (e.g. `from src.engine import RouteOptimizer`).

Keep this file minimal — it should not execute heavy logic on import.
"""

from .heatmap import DeliveryZone, HeatmapGenerator
from .optimizer import Route, RouteOptimizer, Stop
from .scorer import RouteScore, RouteScorer, ScoringWeights

__all__ = [
    "Stop",
    "Route",
    "RouteOptimizer",
    "HeatmapGenerator",
    "DeliveryZone",
    "RouteScorer",
    "ScoringWeights",
    "RouteScore",
]

# Semantic package version (bump when making breaking changes)
__version__ = "0.1.0"


def get_default_engine_components(grid_size_km: float = 1.0):
    """
    Convenience factory that returns a small bundle of default engine components.

    Returns a dict with:
      - optimizer: RouteOptimizer()
      - heatmap: HeatmapGenerator(grid_size_km)
      - scorer: RouteScorer()

    Note: RoutingService is not included here because the routing implementation
    may be optional or not present in all deployments. If you need routing,
    construct it explicitly from src.engine.routing when available.
    """
    optimizer = RouteOptimizer()
    heatmap = HeatmapGenerator(grid_size_km=grid_size_km)
    scorer = RouteScorer()

    return {
        "optimizer": optimizer,
        "heatmap": heatmap,
        "scorer": scorer,
    }
