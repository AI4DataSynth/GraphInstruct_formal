"""Pareto frontier analysis, efficiency ranking, and interactive visualization.

Provides tools for cost-quality trade-off evaluation:
- Pareto frontier computation
- Efficiency metrics (Quality/kToken, Cost@Q, Q@Budget)
- Frontier distance analysis
- Interactive Plotly visualizations (Pareto, radar, heatmap, efficiency)
"""

from graphinstruct.analysis.pareto import (
    ParetoPoint,
    compute_pareto_frontier,
    cost_at_quality,
    frontier_distance,
    is_on_frontier,
    pareto_bonus,
    quality_at_budget,
    quality_per_ktoken,
)

# Visualization imports are optional (plotly may not be installed)
try:
    from graphinstruct.analysis.visualization import (
        efficiency_scatter,
        export_dashboard,
        level_heatmap,
        pareto_frontier_plot,
        radar_comparison,
    )

    _VIZ_EXPORTS = [
        "pareto_frontier_plot",
        "radar_comparison",
        "level_heatmap",
        "efficiency_scatter",
        "export_dashboard",
    ]
except ImportError:
    _VIZ_EXPORTS = []

__all__ = [
    "ParetoPoint",
    "compute_pareto_frontier",
    "cost_at_quality",
    "frontier_distance",
    "is_on_frontier",
    "pareto_bonus",
    "quality_at_budget",
    "quality_per_ktoken",
    *_VIZ_EXPORTS,
]
