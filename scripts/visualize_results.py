"""Visualize benchmark results: bar charts, radar plots, and summary tables.

Usage:
    python scripts/visualize_results.py results/baseline.json
    python scripts/visualize_results.py results/a.json results/b.json --names "LLaMA" "GPT-4"
    python scripts/visualize_results.py results/a.json results/b.json --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REQUIRED_KEYS = {"per_instruction", "per_level_scores", "total_score", "final_score"}


def load_result(path: str) -> dict:
    """Load and validate a benchmark result JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    missing = REQUIRED_KEYS - data.keys()
    if missing:
        print(f"ERROR: {path} missing required keys: {missing}")
        sys.exit(1)
    return data


def _safe_name(name: str) -> str:
    """Sanitize model name for use in filenames."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _collect_level_dimension_scores(data: dict) -> dict[int, dict[str, float]]:
    """Aggregate dimension scores per level (mean)."""
    level_dims: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in data["per_instruction"]:
        lvl = item["level"]
        for dim, val in item.get("dimension_scores", {}).items():
            level_dims[lvl][dim].append(val)

    return {
        lvl: {dim: np.mean(vals) for dim, vals in dims.items()}
        for lvl, dims in level_dims.items()
    }


# ---------------------------------------------------------------------------
# Chart 1: Level scores bar chart
# ---------------------------------------------------------------------------


def plot_level_scores(datasets: list[dict], names: list[str], out_dir: Path) -> None:
    """Bar chart comparing level scores across models."""
    fig, ax = plt.subplots(figsize=(10, 5))

    levels = sorted(set(int(k) for d in datasets for k in d["per_level_scores"].keys()))
    x = np.arange(len(levels))
    width = 0.8 / len(datasets)

    for i, (data, name) in enumerate(zip(datasets, names)):
        scores = [data["per_level_scores"].get(str(lvl), 0.0) for lvl in levels]
        bars = ax.bar(x + i * width, scores, width, label=name, alpha=0.85)
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Instruction Level")
    ax.set_ylabel("Level Score")
    ax.set_title("Level Scores by Instruction Difficulty")
    ax.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels([f"L{lvl}" for lvl in levels])
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "level_scores.png", dpi=150)
    plt.close(fig)
    print("  Saved level_scores.png")


# ---------------------------------------------------------------------------
# Chart 2: Dimension radar chart per level
# ---------------------------------------------------------------------------


def plot_dimension_radar(data: dict, name: str, out_dir: Path) -> None:
    """Radar chart showing D1-D5 scores per level."""
    level_dims = _collect_level_dimension_scores(data)
    dimensions = ["D1", "D2", "D3", "D4", "D5"]
    num_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(level_dims)))
    for (lvl, dims), color in zip(sorted(level_dims.items()), colors):
        values = [dims.get(d, 0.0) for d in dimensions]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.5, label=f"L{lvl}", color=color)
        ax.fill(angles, values, alpha=0.05, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Dimension Scores by Level — {name}", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    safe = _safe_name(name)
    fig.tight_layout()
    fig.savefig(out_dir / f"dimension_radar_{safe}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved dimension_radar_{safe}.png")


# ---------------------------------------------------------------------------
# Chart 3: Per-instruction D1/D4 heatmap
# ---------------------------------------------------------------------------


def plot_instruction_heatmap(data: dict, name: str, out_dir: Path) -> None:
    """Heatmap of D1 (valid rate) and D4 (instruction score) per instruction."""
    items = sorted(
        data["per_instruction"], key=lambda x: (x["level"], x["instruction_id"])
    )

    ids = [it["instruction_id"] for it in items]
    d1 = [it.get("d1_valid_rate", 0.0) for it in items]
    d4 = [it.get("d4_instruction_score", 0.0) for it in items]

    fig_width = min(60, max(12, len(ids) * 0.35))
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 5), sharex=True)

    for ax, values, label, cmap in [
        (axes[0], d1, "D1: Valid Rate", "RdYlGn"),
        (axes[1], d4, "D4: Instruction Score", "RdYlGn"),
    ]:
        arr = np.array(values).reshape(1, -1)
        im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_yticks([0])
        ax.set_yticklabels([label])
        for j, v in enumerate(values):
            ax.text(j, 0, f"{v:.1f}", ha="center", va="center", fontsize=6)

    axes[1].set_xticks(range(len(ids)))
    axes[1].set_xticklabels(ids, rotation=90, fontsize=7)
    fig.suptitle(f"Per-Instruction Scores — {name}", fontsize=12)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Score")

    safe = _safe_name(name)
    fig.tight_layout()
    fig.savefig(out_dir / f"instruction_heatmap_{safe}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved instruction_heatmap_{safe}.png")


# ---------------------------------------------------------------------------
# Chart 4: Pareto frontier (quality vs cost)
# ---------------------------------------------------------------------------


def plot_pareto_frontier(datasets: list[dict], names: list[str], out_dir: Path) -> None:
    """Scatter plot of quality (total_score) vs cost (tokens per valid graph).

    Highlights the Pareto frontier connecting non-dominated points.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ["o", "s", "D", "^", "v", "P", "*", "X"]

    all_x: list[float] = []
    all_y: list[float] = []

    for i, (data, name) in enumerate(zip(datasets, names)):
        # Compute average tokens per valid graph
        total_tokens = 0
        valid_count = 0
        for item in data["per_instruction"]:
            tokens_per_valid = item.get("d5_tokens_per_valid")
            if tokens_per_valid is not None:
                total_tokens += tokens_per_valid
                valid_count += 1
        avg_cost = total_tokens / valid_count if valid_count > 0 else 0
        quality = data["total_score"]

        marker = markers[i % len(markers)]
        ax.scatter(avg_cost, quality, s=120, marker=marker, label=name, zorder=3)
        all_x.append(avg_cost)
        all_y.append(quality)

    # Draw Pareto frontier if 2+ points
    if len(all_x) >= 2:
        points = sorted(zip(all_x, all_y), key=lambda p: p[0])
        frontier_x = [points[0][0]]
        frontier_y = [points[0][1]]
        best_y = points[0][1]
        for px, py in points[1:]:
            if py >= best_y:
                frontier_x.append(px)
                frontier_y.append(py)
                best_y = py
        if len(frontier_x) >= 2:
            ax.plot(
                frontier_x,
                frontier_y,
                "r--",
                linewidth=1.5,
                alpha=0.7,
                label="Pareto Frontier",
            )

    ax.set_xlabel("Avg Tokens per Valid Graph (Cost)")
    ax.set_ylabel("Total Score (Quality)")
    ax.set_title("Pareto Frontier: Quality vs Cost")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "pareto_frontier.png", dpi=150)
    plt.close(fig)
    print("  Saved pareto_frontier.png")


# ---------------------------------------------------------------------------
# Chart 5: Summary table (text)
# ---------------------------------------------------------------------------


def print_summary_table(datasets: list[dict], names: list[str]) -> None:
    """Print a text summary table."""
    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)

    header = f"{'Model':<20} {'S_total':>8} {'S_final':>8}"
    levels = sorted(set(int(k) for d in datasets for k in d["per_level_scores"].keys()))
    for lvl in levels:
        header += f" {'L' + str(lvl):>6}"
    print(header)
    print("-" * len(header))

    for data, name in zip(datasets, names):
        row = f"{name:<20} {data['total_score']:>8.4f} {data['final_score']:>8.4f}"
        for lvl in levels:
            score = data["per_level_scores"].get(str(lvl), 0.0)
            row += f" {score:>6.3f}"
        print(row)

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_interactive_results(datasets: list[dict], names: list[str]) -> dict:
    """Convert list of datasets into the multi-model format expected by
    the interactive visualization module."""
    return {"models": {name: data for name, data in zip(names, datasets)}}


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("results", nargs="+", help="Result JSON file(s)")
    parser.add_argument("--names", nargs="+", help="Model names (one per result file)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive HTML dashboard (requires plotly)",
    )
    args = parser.parse_args()

    # Load with error handling
    datasets = []
    for p in args.results:
        try:
            datasets.append(load_result(p))
        except FileNotFoundError:
            print(f"ERROR: File not found: {p}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in {p}: {e}")
            sys.exit(1)

    names = args.names or [Path(p).stem for p in args.results]
    if len(names) != len(datasets):
        print("ERROR: --names count must match number of result files")
        sys.exit(1)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    if args.interactive:
        # Interactive mode: generate HTML dashboard with plotly
        try:
            from graphinstruct.analysis.visualization import export_dashboard
        except ImportError:
            print("ERROR: plotly is required for --interactive mode.")
            print("Install with: pip install plotly")
            sys.exit(1)

        multi_results = _build_interactive_results(datasets, names)
        html_path = export_dashboard(multi_results, str(out_dir / "dashboard.html"))
        print(f"  Interactive dashboard saved to {html_path}")
    else:
        # Static mode: generate matplotlib PNG charts
        plot_level_scores(datasets, names, out_dir)
        for data, name in zip(datasets, names):
            plot_dimension_radar(data, name, out_dir)
            plot_instruction_heatmap(data, name, out_dir)
        plot_pareto_frontier(datasets, names, out_dir)

    print_summary_table(datasets, names)
    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
