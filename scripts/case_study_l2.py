"""Render the L2-143 capability-gap case study.

Three side-by-side panels show the same instruction handled by:
  (a) Reference (algorithmic ground truth)
  (b) Sonnet-4.6 zero-shot   (frontier, T1)
  (c) GPT-4o-mini zero-shot  (small, T3 capability floor)

Each panel includes the rendered graph plus a constraint checklist
indicating which explicit and implicit constraints are satisfied.
The figure supports the RQ1 finding that L2 multi-constraint
composition is the point of maximal tier discrimination.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from palette import (  # noqa: E402
    ANNOTATION_COLOR,
    HIGHLIGHT_COLOR,
    PARETO_COLOR,
)

INSTRUCTION_FILE = PROJECT_ROOT / "data" / "instructions" / "level_2.json"
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = PROJECT_ROOT / "paper-neurips-v2" / "figures"

CASE_ID = "L2-143"


# ---------------------------------------------------------------------------
# Parse the InstructGraph code-style format into a NetworkX graph
# ---------------------------------------------------------------------------
def parse_code_style(serialized: str) -> nx.Graph:
    g = nx.Graph()
    nodes_match = re.search(r"node_list\s*=\s*\[(.*?)\]", serialized, re.DOTALL)
    edges_match = re.search(r"edge_list\s*=\s*\[(.*?)\]", serialized, re.DOTALL)
    if nodes_match:
        for tok in re.findall(r"'([^']+)'", nodes_match.group(1)):
            g.add_node(tok)
    if edges_match:
        edge_block = edges_match.group(1)
        for u, v in re.findall(r"\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)", edge_block):
            g.add_edge(u, v)
    return g


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------
def grid_layout_3x5_int(g: nx.Graph) -> dict[str, tuple[float, float]]:
    """For the reference: nodes 0-4 row 0, 5-9 row 1, 10-14 row 2."""
    pos: dict[str, tuple[float, float]] = {}
    for node in g.nodes:
        try:
            n = int(node)
        except ValueError:
            continue
        row = n // 5
        col = n % 5
        pos[node] = (col, -row)
    return pos


def grid_layout_3x5_coord(g: nx.Graph) -> dict[str, tuple[float, float]]:
    """For Sonnet: nodes labelled '00','01','02','03','04','10',...,'24'."""
    pos: dict[str, tuple[float, float]] = {}
    for node in g.nodes:
        if len(node) == 2 and node.isdigit():
            row, col = int(node[0]), int(node[1])
            pos[node] = (col, -row)
    return pos


def spring_layout(g: nx.Graph) -> dict:
    """Spring layout rescaled to [0, 0.9] in y so panel title has clear room."""
    raw = nx.spring_layout(g, seed=7, k=0.7, iterations=100)
    if not raw:
        return raw
    xs = [p[0] for p in raw.values()]
    ys = [p[1] for p in raw.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    scaled: dict[str, tuple[float, float]] = {}
    for node, (x, y) in raw.items():
        # Rescale x to [0, 1] and y to [0, 0.85] so top of panel stays clear.
        nx_v = (x - x_min) / x_span
        ny_v = (y - y_min) / y_span * 0.85
        scaled[node] = (nx_v, ny_v)
    return scaled


# ---------------------------------------------------------------------------
# Constraint checking
# ---------------------------------------------------------------------------
def evaluate_constraints(g: nx.Graph, target_nodes: int = 15) -> list[tuple[str, bool, str]]:
    """Return [(constraint_label, satisfied, observed_value), ...]."""
    deg = [d for _, d in g.degree()]
    min_deg = min(deg) if deg else 0
    is_connected = nx.is_connected(g) if g.number_of_nodes() > 0 else False
    try:
        is_planar, _ = nx.check_planarity(g)
    except Exception:
        is_planar = False
    is_bipartite = nx.is_bipartite(g)
    is_grid = (
        g.number_of_nodes() == 15
        and g.number_of_edges() == 22
        and is_connected
        and is_planar
        and is_bipartite
        and min_deg >= 2
    )
    return [
        ("graph_type = grid", is_grid, "structural match" if is_grid else "broken layout"),
        ("num_nodes = 15", g.number_of_nodes() == 15, str(g.number_of_nodes())),
        ("num_edges = 22", g.number_of_edges() == 22, str(g.number_of_edges())),
        ("connected", is_connected, "yes" if is_connected else "no"),
        ("planar", is_planar, "yes" if is_planar else "no"),
        ("bipartite", is_bipartite, "yes" if is_bipartite else "no"),
        ("min_degree >= 2", min_deg >= 2, f"min={min_deg}"),
    ]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_panel(
    ax,
    g: nx.Graph,
    pos: dict,
    title: str,
    subtitle: str,
    is_reference: bool,
    constraints: Sequence[tuple[str, bool, str]],
    target_nodes: set[str] | None = None,
) -> None:
    # Color nodes: highlight extra/wrong nodes in red, kept nodes in palette green
    node_colors = []
    for n in g.nodes:
        if target_nodes is not None and n not in target_nodes:
            node_colors.append(HIGHLIGHT_COLOR)
        else:
            node_colors.append(PARETO_COLOR if is_reference else "#7589B7")

    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#666666", width=1.0)
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=node_colors, node_size=380, edgecolors="black", linewidths=0.8
    )
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8, font_color="white")
    ax.margins(x=0.12, y=0.20)

    # Outcome banner
    n_violations = sum(1 for _, ok, _ in constraints if not ok)
    if n_violations == 0:
        outcome = "ALL CONSTRAINTS SATISFIED"
        outcome_color = PARETO_COLOR
    else:
        outcome = f"{n_violations} CONSTRAINT(S) VIOLATED"
        outcome_color = HIGHLIGHT_COLOR
    # Use set_title() so matplotlib positions the title in figure space rather
    # than in axes-relative space, keeping it anchored to the panel header
    # regardless of whether the underlying graph layout is wide or square.
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, pad=10, loc="center")
    # Outcome banner under the panel via a low-anchored text in figure-relative
    # axes coords; this consistently sits below the rendered nodes because
    # we leave 20% bottom margin.
    ax.text(
        0.5,
        -0.08,
        outcome,
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=outcome_color,
    )
    ax.set_axis_off()


def draw_constraints_table(ax, constraints: Sequence[tuple[str, bool, str]], title: str) -> None:
    ax.set_axis_off()
    ax.text(
        0.0,
        1.04,
        title,
        transform=ax.transAxes,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    line_height = 0.11
    for i, (label, ok, observed) in enumerate(constraints):
        y = 0.92 - i * line_height
        marker = "✓" if ok else "✗"  # check / cross
        marker_color = PARETO_COLOR if ok else HIGHLIGHT_COLOR
        ax.text(0.02, y, marker, transform=ax.transAxes, fontsize=14, color=marker_color, fontweight="bold")
        ax.text(0.10, y, label, transform=ax.transAxes, fontsize=10, color=ANNOTATION_COLOR)
        ax.text(0.74, y, observed, transform=ax.transAxes, fontsize=9, color=marker_color, style="italic")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_instruction(case_id: str) -> dict:
    with open(INSTRUCTION_FILE, encoding="utf-8") as f:
        data = json.load(f)
    for inst in data:
        if inst["id"] == case_id:
            return inst
    raise KeyError(case_id)


def load_first_sample(stem: str, case_id: str) -> dict:
    with open(RESULTS_DIR / f"{stem}.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["instruction_id"] == case_id:
                return rec["samples"][0]
    raise KeyError(case_id)


def main() -> None:
    inst = load_instruction(CASE_ID)
    ref_serialized = inst["reference_solutions"][0]
    son_sample = load_first_sample("claudesonnet46-zero-shot", CASE_ID)
    gpt_sample = load_first_sample("gpt4omini-zero-shot", CASE_ID)

    g_ref = parse_code_style(ref_serialized)
    g_son = parse_code_style(son_sample["graph_serialized"])
    g_gpt = parse_code_style(gpt_sample["graph_serialized"])

    pos_ref = grid_layout_3x5_int(g_ref)
    pos_son = grid_layout_3x5_coord(g_son)
    pos_gpt = spring_layout(g_gpt)

    # Identify extra nodes in GPT output (target = nodes 0-14)
    target_nodes_gpt = {str(i) for i in range(15)}

    cons_ref = evaluate_constraints(g_ref)
    cons_son = evaluate_constraints(g_son)
    cons_gpt = evaluate_constraints(g_gpt)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), gridspec_kw={"height_ratios": [3, 1.6]})

    # Top row: graphs
    draw_panel(
        axes[0, 0],
        g_ref,
        pos_ref,
        "(a) Reference (ground truth)",
        "algorithmically synthesized",
        is_reference=True,
        constraints=cons_ref,
    )
    draw_panel(
        axes[0, 1],
        g_son,
        pos_son,
        "(b) Sonnet-4.6 (frontier, T1)",
        "Quality = 0.859 – satisfies the spec",
        is_reference=False,
        constraints=cons_son,
    )
    draw_panel(
        axes[0, 2],
        g_gpt,
        pos_gpt,
        "(c) GPT-4o-mini (small, T3)",
        "Quality = 0.000 – 19 nodes / 26 edges (target: 15 / 22)",
        is_reference=False,
        constraints=cons_gpt,
        target_nodes=target_nodes_gpt,
    )

    # Bottom row: constraint checklists
    draw_constraints_table(axes[1, 0], cons_ref, "Constraints (target)")
    draw_constraints_table(axes[1, 1], cons_son, "Sonnet-4.6 verdict")
    draw_constraints_table(axes[1, 2], cons_gpt, "GPT-4o-mini verdict")

    fig.suptitle(
        f"Case study (L2-143): “{inst['instruction']}”",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    out = OUT_DIR / "fig_case_study_L2.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
