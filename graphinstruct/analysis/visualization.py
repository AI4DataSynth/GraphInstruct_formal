"""Interactive visualization for GraphInstruct benchmark results.

Generates interactive Plotly HTML charts for Pareto frontiers,
radar comparisons, level heatmaps, and efficiency scatter plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from graphinstruct.analysis.pareto import (
    ParetoPoint,
    compute_pareto_frontier,
    is_on_frontier,
)

# Unified color palette for models
MODEL_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]

DIMENSION_LABELS = {
    "D1": "D1 Structural",
    "D2": "D2 Textual",
    "D3": "D3 Embedding",
    "D4": "D4 Instruction Match",
    "D5": "D5 Token Efficiency",
}

DIMENSIONS = ["D1", "D2", "D3", "D4", "D5"]


def _require_plotly() -> None:
    """Raise ImportError if plotly is not available."""
    if not HAS_PLOTLY:
        raise ImportError(
            "plotly is required for interactive visualization. "
            "Install it with: pip install plotly"
        )


def _extract_models(results: Dict[str, Any]) -> List[str]:
    """Extract model names from a multi-model results dict."""
    return list(results.get("models", {}).keys())


def _get_color(index: int) -> str:
    """Get a color from the palette by index."""
    return MODEL_COLORS[index % len(MODEL_COLORS)]


# ---------------------------------------------------------------------------
# Pareto frontier plot
# ---------------------------------------------------------------------------


def pareto_frontier_plot(
    results: Dict[str, Any],
    x_metric: str = "total_tokens",
    y_metric: str = "s_total",
) -> "go.Figure":
    """Interactive Pareto frontier scatter plot.

    Args:
        results: Dict with 'models' key mapping model_name -> model_data.
            Each model_data has 'per_instruction', 'total_score',
            'final_score', 'per_level_scores'.
        x_metric: Cost metric for x-axis ('total_tokens').
        y_metric: Quality metric for y-axis ('s_total').

    Returns:
        Plotly Figure object.
    """
    _require_plotly()

    models = results.get("models", {})
    if not models:
        raise ValueError("results must contain a 'models' dict with model data")

    fig = go.Figure()

    pareto_points: list[ParetoPoint] = []
    model_names = list(models.keys())

    for idx, (name, data) in enumerate(models.items()):
        cost = _compute_avg_cost(data)
        quality = data.get("total_score", 0.0)
        valid_rate = _compute_valid_rate(data)

        pareto_points.append(
            ParetoPoint(label=name, quality=quality, cost=cost, valid_rate=valid_rate)
        )

        # Hover text with details
        hover = (
            f"<b>{name}</b><br>"
            f"Quality (S_total): {quality:.4f}<br>"
            f"Token cost: {cost:.0f}<br>"
            f"Valid rate: {valid_rate:.1%}<br>"
            f"Final score: {data.get('final_score', 0.0):.4f}"
        )

        fig.add_trace(
            go.Scatter(
                x=[cost],
                y=[quality],
                mode="markers+text",
                marker=dict(
                    size=max(12, valid_rate * 30),
                    color=_get_color(idx),
                    line=dict(width=2, color="white"),
                ),
                text=[name],
                textposition="top center",
                hovertemplate=hover + "<extra></extra>",
                name=name,
            )
        )

    # Draw Pareto frontier line
    frontier = compute_pareto_frontier(pareto_points)
    if len(frontier) >= 2:
        f_costs = [p.cost for p in frontier]
        f_quals = [p.quality for p in frontier]
        fig.add_trace(
            go.Scatter(
                x=f_costs,
                y=f_quals,
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                name="Pareto frontier",
                hoverinfo="skip",
            )
        )

    # Highlight frontier points
    for pt in frontier:
        fig.add_trace(
            go.Scatter(
                x=[pt.cost],
                y=[pt.quality],
                mode="markers",
                marker=dict(
                    size=18,
                    color="rgba(255,0,0,0)",
                    line=dict(width=3, color="red"),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Pareto frontier: Quality vs Token cost",
        xaxis_title="Average Token cost",
        yaxis_title="Total Quality score (S_total)",
        hovermode="closest",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


# ---------------------------------------------------------------------------
# Radar comparison
# ---------------------------------------------------------------------------


def radar_comparison(
    results: Dict[str, Any],
    models: Optional[List[str]] = None,
) -> "go.Figure":
    """Interactive 5D radar chart comparing models on D1-D5.

    Args:
        results: Dict with 'models' key.
        models: Optional list of model names to include.
            If None, all models are shown.

    Returns:
        Plotly Figure object.
    """
    _require_plotly()

    all_models = results.get("models", {})
    if not all_models:
        raise ValueError("results must contain a 'models' dict")

    selected = models if models else list(all_models.keys())

    fig = go.Figure()

    categories = [DIMENSION_LABELS[d] for d in DIMENSIONS]

    for idx, name in enumerate(selected):
        if name not in all_models:
            continue
        data = all_models[name]
        dim_scores = _compute_avg_dimension_scores(data)

        values = [dim_scores.get(d, 0.0) for d in DIMENSIONS]
        # Close the radar polygon
        values_closed = values + [values[0]]
        cats_closed = categories + [categories[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=cats_closed,
                fill="toself",
                fillcolor=_get_color(idx).replace(")", ", 0.1)").replace("rgb", "rgba"),
                line=dict(color=_get_color(idx), width=2),
                name=name,
                hovertemplate=(
                    f"<b>{name}</b><br>" "%{theta}: %{r:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        title="Five-dimension Radar Chart (D1-D5)",
        template="plotly_white",
        showlegend=True,
    )

    return fig


# ---------------------------------------------------------------------------
# Level heatmap
# ---------------------------------------------------------------------------


def level_heatmap(results: Dict[str, Any]) -> "go.Figure":
    """Interactive heatmap: models x instruction levels.

    Args:
        results: Dict with 'models' key.

    Returns:
        Plotly Figure object.
    """
    _require_plotly()

    all_models = results.get("models", {})
    if not all_models:
        raise ValueError("results must contain a 'models' dict")

    model_names = list(all_models.keys())

    # Collect all levels
    all_levels: set[int] = set()
    for data in all_models.values():
        for k in data.get("per_level_scores", {}).keys():
            all_levels.add(int(k))
    levels = sorted(all_levels)

    # Build matrix: rows=models, cols=levels
    z: list[list[float]] = []
    hover_text: list[list[str]] = []

    for name in model_names:
        data = all_models[name]
        pls = data.get("per_level_scores", {})
        row: list[float] = []
        hover_row: list[str] = []
        for lvl in levels:
            score = pls.get(str(lvl), 0.0)
            row.append(score)
            hover_row.append(f"<b>{name}</b><br>" f"Level {lvl}: {score:.4f}")
        z.append(row)
        hover_text.append(hover_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[f"L{lvl}" for lvl in levels],
            y=model_names,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=[[f"{v:.3f}" for v in row] for row in z],
            texttemplate="%{text}",
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            colorbar=dict(title="Score"),
        )
    )

    fig.update_layout(
        title="Model x Instruction Level Score Heatmap",
        xaxis_title="Instruction Level",
        yaxis_title="Model",
        template="plotly_white",
    )

    return fig


# ---------------------------------------------------------------------------
# Efficiency scatter
# ---------------------------------------------------------------------------


def efficiency_scatter(results: Dict[str, Any]) -> "go.Figure":
    """Quality vs Token efficiency scatter plot.

    Args:
        results: Dict with 'models' key.

    Returns:
        Plotly Figure object.
    """
    _require_plotly()

    all_models = results.get("models", {})
    if not all_models:
        raise ValueError("results must contain a 'models' dict")

    fig = go.Figure()
    pareto_points: list[ParetoPoint] = []

    for idx, (name, data) in enumerate(all_models.items()):
        cost = _compute_avg_cost(data)
        quality = data.get("total_score", 0.0)
        valid_rate = _compute_valid_rate(data)

        # Size based on valid_rate, color by model
        marker_size = max(15, valid_rate * 50)

        pareto_points.append(
            ParetoPoint(label=name, quality=quality, cost=cost, valid_rate=valid_rate)
        )

        hover = (
            f"<b>{name}</b><br>"
            f"Quality: {quality:.4f}<br>"
            f"Token cost: {cost:.0f}<br>"
            f"Valid rate: {valid_rate:.1%}<br>"
            f"Quality/kToken: {quality / max(cost / 1000, 0.001):.4f}"
        )

        fig.add_trace(
            go.Scatter(
                x=[cost],
                y=[quality],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=_get_color(idx),
                    opacity=0.8,
                    line=dict(width=1, color="white"),
                ),
                name=name,
                hovertemplate=hover + "<extra></extra>",
            )
        )

    # Overlay Pareto frontier
    frontier = compute_pareto_frontier(pareto_points)
    if len(frontier) >= 2:
        f_costs = [p.cost for p in frontier]
        f_quals = [p.quality for p in frontier]
        fig.add_trace(
            go.Scatter(
                x=f_costs,
                y=f_quals,
                mode="lines",
                line=dict(color="rgba(255,0,0,0.5)", width=2, dash="dot"),
                name="Pareto frontier",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Quality vs Token Efficiency",
        xaxis_title="Average Token cost",
        yaxis_title="Total Quality score (S_total)",
        hovermode="closest",
        template="plotly_white",
    )

    return fig


# ---------------------------------------------------------------------------
# Dashboard export
# ---------------------------------------------------------------------------


def export_dashboard(
    results: Dict[str, Any],
    output_path: str,
) -> str:
    """Generate a single self-contained HTML dashboard with all plots.

    Args:
        results: Multi-model results dict.
        output_path: Path to write the HTML file.

    Returns:
        Absolute path to the generated HTML file.
    """
    _require_plotly()

    # Generate all plots
    figs = {
        "pareto": pareto_frontier_plot(results),
        "radar": radar_comparison(results),
        "heatmap": level_heatmap(results),
        "efficiency": efficiency_scatter(results),
    }

    # Build combined HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        '<meta charset="utf-8">',
        "<title>GraphInstruct Benchmark Dashboard</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; "
        "background: #fafafa; }",
        "h1 { text-align: center; color: #333; }",
        ".chart-container { margin: 20px auto; max-width: 1200px; "
        "background: white; padding: 20px; border-radius: 8px; "
        "box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "</style>",
        "</head><body>",
        "<h1>GraphInstruct Benchmark Dashboard</h1>",
    ]

    for title, fig in figs.items():
        inner_html = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn" if title != "pareto" else True,
        )
        html_parts.append(f'<div class="chart-container">{inner_html}</div>')

    html_parts.append("</body></html>")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(html_parts), encoding="utf-8")

    return str(out.resolve())


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compute_avg_cost(data: Dict[str, Any]) -> float:
    """Compute average tokens per valid graph from per_instruction data."""
    total_tokens = 0.0
    count = 0
    for item in data.get("per_instruction", []):
        tpv = item.get("d5_tokens_per_valid")
        if tpv is not None and tpv != float("inf"):
            total_tokens += tpv
            count += 1
    return total_tokens / count if count > 0 else 0.0


def _compute_valid_rate(data: Dict[str, Any]) -> float:
    """Compute average valid rate from per_instruction data."""
    rates = [item.get("d1_valid_rate", 0.0) for item in data.get("per_instruction", [])]
    return sum(rates) / len(rates) if rates else 0.0


def _compute_avg_dimension_scores(data: Dict[str, Any]) -> Dict[str, float]:
    """Compute average dimension scores across all instructions."""
    totals: Dict[str, float] = {d: 0.0 for d in DIMENSIONS}
    count = 0
    for item in data.get("per_instruction", []):
        ds = item.get("dimension_scores", {})
        if ds:
            for d in DIMENSIONS:
                totals[d] += ds.get(d, 0.0)
            count += 1
    if count == 0:
        return totals
    return {d: v / count for d, v in totals.items()}
