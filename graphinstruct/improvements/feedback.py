"""Layered feedback text generation for VGIG iteration.

Converts ``ViolationDetail`` lists into natural-language feedback text the
LLM can act on in the next round. Supports three granularity levels
(``none`` / ``coarse`` / ``fine``) for the E6 ablation study.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx

from graphinstruct.data_loader import Instruction
from graphinstruct.improvements.violation import (
    ConstraintCategory,
    ViolationDetail,
    _split_key,
    extract_violations,
)

FeedbackLevel = Literal["none", "coarse", "fine"]


# Priority ordering: hard structural failures must be fixed before downstream
# numerical properties can settle.
_PRIORITY: dict[ConstraintCategory, int] = {
    "structural": 0,
    "directed": 1,
    "count": 2,
    "degree": 3,
    "numerical": 4,
    "domain": 5,
    "unknown": 6,
}

_DEFAULT_MAX_VIOLATIONS = 8


_NUMERICAL_HINTS: dict[tuple[str, str], str] = {
    ("density", "higher"): "remove some edges to reduce density",
    ("density", "lower"): "add more edges to increase density",
    ("clustering", "higher"): "spread edges more evenly (fewer triangles)",
    ("clustering", "lower"): "add triangles between neighbours",
    ("clustering_coefficient", "higher"): "spread edges more evenly (fewer triangles)",
    ("clustering_coefficient", "lower"): "add triangles between neighbours",
    ("avg_path_length", "higher"): "add shortcut edges between distant nodes",
    ("avg_path_length", "lower"): "remove direct shortcuts to lengthen paths",
    ("average_path_length", "higher"): "add shortcut edges between distant nodes",
    ("average_path_length", "lower"): "remove direct shortcuts to lengthen paths",
    ("diameter", "higher"): "add long-range shortcut edges",
    ("diameter", "lower"): "remove edges that shorten the diameter",
    (
        "modularity",
        "higher",
    ): "strengthen community structure (cluster edges within groups)",
    ("modularity", "lower"): "spread edges across community boundaries",
    ("node_connectivity", "lower"): "add more internal edges between parts",
    ("edge_connectivity", "lower"): "add bridging edges between weakly connected areas",
}


def generate_feedback(
    graph: nx.Graph | nx.DiGraph | None,
    instruction: Instruction,
    *,
    level: FeedbackLevel = "fine",
    max_violations: int = _DEFAULT_MAX_VIOLATIONS,
) -> str:
    """Produce feedback text for VGIG round-t prompt augmentation.

    Returns an empty string when all constraints are satisfied, signalling
    convergence to the runner (which will stop iterating).
    """
    violations = extract_violations(
        graph,
        instruction.explicit_constraints,
        instruction.implicit_constraints,
        getattr(instruction, "domain", None),
    )
    unsatisfied = [v for v in violations if not v.satisfied]
    if not unsatisfied:
        return ""

    if level == "none":
        return _format_none(unsatisfied)
    if level == "coarse":
        return _format_coarse(unsatisfied)
    if level == "fine":
        return _format_fine(unsatisfied, max_violations)
    raise ValueError(f"Unknown feedback level: {level!r}")


# ---------------------------------------------------------------------------
# Private formatters
# ---------------------------------------------------------------------------


def _format_none(_: list[ViolationDetail]) -> str:
    """Fixed no-info feedback for the E6 ablation's lowest tier."""
    return (
        "Your previous graph did not satisfy all constraints. "
        "Please try again in the same format."
    )


def _format_coarse(violations: list[ViolationDetail]) -> str:
    """List only the names of violated constraints, no diagnostic detail."""
    keys: list[str] = []
    seen: set[str] = set()
    for v in violations:
        if v.constraint == "parse":
            key = "parseable_output"
        else:
            key = _split_key(v.constraint) or v.constraint
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return (
        "Your previous graph failed the following constraints: "
        + ", ".join(keys)
        + ". Please regenerate addressing these issues."
    )


def _format_fine(violations: list[ViolationDetail], max_violations: int) -> str:
    """Full diagnostic text grouped by priority."""
    sorted_v = sorted(violations, key=lambda v: _PRIORITY.get(v.category, 99))
    truncated = sorted_v[:max_violations]

    lines = ["Your previous graph did not satisfy all constraints. Issues found:"]
    for v in truncated:
        if v.category == "structural":
            lines.append("- " + _format_structural(v))
        elif v.category == "degree":
            lines.append("- " + _format_degree(v))
        elif v.category == "count":
            lines.append("- " + _format_count(v))
        elif v.category == "directed":
            lines.append("- " + _format_directed(v))
        elif v.category == "numerical":
            lines.append("- " + _format_numerical(v))
        elif v.category == "domain":
            lines.append("- " + _format_domain(v))
        else:
            lines.append(f"- {v.constraint} was violated")

    if len(sorted_v) > max_violations:
        lines.append(f"(... and {len(sorted_v) - max_violations} more)")

    lines.append("Please regenerate the graph in the same format, fixing these issues.")
    return "\n".join(lines)


def _format_structural(v: ViolationDetail) -> str:
    if v.constraint == "parse":
        return (
            "Your previous output could not be parsed. "
            "Regenerate a valid graph using the required format."
        )
    key = (_split_key(v.constraint) or "").lower()

    if key == "graph_type":
        pieces: list[str] = []
        loc = v.locus or {}
        if v.expected == "tree" and loc:
            if loc.get("num_edges_actual") != loc.get("num_edges_expected"):
                pieces.append(
                    f"edges: got {loc.get('num_edges_actual')}, "
                    f"need {loc.get('num_edges_expected')}"
                )
            if loc.get("connected") is False:
                pieces.append("graph is not connected")
            if loc.get("acyclic") is False:
                pieces.append("graph contains cycles")
        elif loc:
            if "num_nodes" in loc:
                pieces.append(f"num_nodes={loc['num_nodes']}")
            if "num_edges" in loc:
                pieces.append(f"num_edges={loc['num_edges']}")
        hint = " (" + "; ".join(pieces) + ")" if pieces else ""
        return f"graph_type should be {v.expected}{hint}"

    if key == "connected":
        components = (v.locus or {}).get("components", [])
        actual_desc = v.actual or f"{len(components)} components"
        sample = components[:3]
        return (
            f"The graph must be connected (currently {actual_desc}; "
            f"sample components: {sample})"
        )

    if key == "acyclic":
        cycles = (v.locus or {}).get("cycles", [])
        if cycles:
            return f"The graph must be acyclic; found cycle: {cycles[0]}"
        return "The graph must be acyclic; remove at least one edge from each cycle"

    if key == "bipartite":
        return "The graph must be bipartite (no odd-length cycles)"

    if key == "planar":
        return "The graph must be planar (drawable without edge crossings)"

    if key in ("chromatic_number", "girth"):
        return f"{key} should satisfy {v.constraint} (currently {v.actual})"

    return f"Structural constraint {v.constraint} violated"


def _format_degree(v: ViolationDetail) -> str:
    key = (_split_key(v.constraint) or "").lower()
    loc = v.locus or {}
    offenders = loc.get("offending_nodes", [])
    table = loc.get("degree_table", {})
    count = loc.get("offender_count", len(offenders))
    offender_tail = (
        f" (total {count} offending nodes)" if count > len(offenders) else ""
    )

    if key == "degree":
        if v.actual == "non-uniform":
            sample = ", ".join(f"node {n}: deg {d}" for n, d in list(table.items())[:5])
            return (
                f"All nodes must have degree {v.expected} (currently non-uniform; "
                f"sample: {sample}){offender_tail}"
            )
        if loc.get("empty_graph"):
            return f"All nodes must have degree {v.expected} (graph is currently empty)"
        return (
            f"All nodes must have degree {v.expected} (actual uniform degree {v.actual}; "
            f"offending nodes {offenders[:5]}){offender_tail}"
        )
    if key == "min_degree":
        return (
            f"Minimum degree must be >= {v.expected} (current min {v.actual}; "
            f"nodes below threshold {offenders[:5]}){offender_tail}"
        )
    if key == "max_degree":
        return (
            f"Maximum degree must be <= {v.expected} (current max {v.actual}; "
            f"nodes above threshold {offenders[:5]}){offender_tail}"
        )
    return f"Degree constraint {v.constraint} violated"


def _format_count(v: ViolationDetail) -> str:
    key = (_split_key(v.constraint) or "").lower()
    delta = (v.locus or {}).get("need_delta", 0)
    if delta > 0:
        action = f"add {delta} more"
    elif delta < 0:
        action = f"remove {-delta}"
    else:
        action = "no change needed"
    if key == "num_nodes":
        return f"Graph has {v.actual} nodes but requires {v.expected} ({action})"
    if key == "num_edges":
        return f"Graph has {v.actual} edges but requires {v.expected} ({action})"
    return f"Count constraint {v.constraint} violated"


def _format_directed(v: ViolationDetail) -> str:
    return f"Graph should be {v.expected} but is {v.actual}"


def _format_numerical(v: ViolationDetail) -> str:
    key = (_split_key(v.constraint) or "").lower()
    loc = v.locus or {}
    direction = loc.get("direction", "unknown")
    gap = loc.get("gap")

    def _fmt(x: object) -> str:
        if isinstance(x, float):
            return f"{x:.3f}"
        return str(x)

    actual_str = _fmt(v.actual)
    expected_str = _fmt(v.expected)
    gap_str = f" (gap {gap:.3f})" if isinstance(gap, float) else ""
    hint = _numerical_hint(key, direction) if direction in ("higher", "lower") else ""
    hint_str = f" -> {hint}" if hint else ""
    return f"{key} is {actual_str}, target {expected_str}{gap_str}{hint_str}"


def _format_domain(v: ViolationDetail) -> str:
    loc = v.locus or {}
    domain = loc.get("domain", "unknown")
    mismatches = loc.get("mismatch", [])
    if not mismatches:
        return f"Domain {domain}: all structural statistics within expected range"
    pieces = [f"{name}={actual} (expected {rng})" for (name, actual, rng) in mismatches]
    return f"Domain {domain} mismatch: " + "; ".join(pieces)


def _numerical_hint(constraint_key: str, direction: Literal["higher", "lower"]) -> str:
    """Short actionable hint keyed on (property, direction)."""
    return _NUMERICAL_HINTS.get((constraint_key, direction), "")
