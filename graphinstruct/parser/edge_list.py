"""Parser for edge list graph format.

Format example:
    Nodes: 0, 1, 2, 3, 4
    Edges:
    0 1
    0 2
    1 3
    3 4
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import networkx as nx

from graphinstruct.parser.code_style import ParseResult


def parse(text: str) -> ParseResult:
    """Parse an edge list format string into a ParseResult.

    Supports formats:
    - "0 1" (space separated pair)
    - "0, 1" (comma separated pair)
    - "(0, 1)" (tuple-style)
    - Optional "Nodes:" header line
    - Optional "Edges:" header line

    Returns:
        ParseResult with the parsed graph and metadata.

    Raises:
        ValueError: If the format cannot be parsed.
    """
    text = text.strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Detect directed
    is_directed = _detect_directed(lines)
    graph_cls = nx.DiGraph if is_directed else nx.Graph

    node_list: list[str] = []
    edge_lines: list[str] = []
    in_edge_section = False

    for line in lines:
        lower = line.lower()
        # Skip comment lines
        if line.startswith("#"):
            continue
        if lower.startswith("directed"):
            continue
        if lower.startswith("nodes:"):
            node_part = line.split(":", 1)[1].strip()
            node_list = [
                n.strip().strip("'\"") for n in node_part.split(",") if n.strip()
            ]
            continue
        if lower.startswith("edge"):
            in_edge_section = True
            continue

        # Try parsing as edge
        edge = _parse_edge_line(line)
        if edge is not None:
            edge_lines.append(line)
            in_edge_section = True

    if not edge_lines and not node_list:
        raise ValueError("No edges or nodes found in edge list format")

    G = graph_cls()
    if node_list:
        G.add_nodes_from(node_list)

    for line in edge_lines:
        edge = _parse_edge_line(line)
        if edge is not None:
            u, v = edge
            G.add_node(u)
            G.add_node(v)
            G.add_edge(u, v)

    return ParseResult(
        parse_type="Graph",
        metadata={"format": "edge_list"},
        graph=G,
    )


def serialize(
    graph: nx.Graph,
    name: str = "",
    **extra_metadata: Any,
) -> str:
    """Serialize a NetworkX graph to edge list format.

    Args:
        graph: The NetworkX graph to serialize.
        name: Optional graph name (included as comment).

    Returns:
        Edge list format string.
    """
    lines: list[str] = []
    is_directed = isinstance(graph, nx.DiGraph)

    if name:
        lines.append(f"# {name}")
    if is_directed:
        lines.append("Directed: true")

    nodes = sorted(graph.nodes(), key=str)
    lines.append(f"Nodes: {', '.join(str(n) for n in nodes)}")
    lines.append("Edges:")

    for u, v in sorted(graph.edges()):
        lines.append(f"{u} {v}")

    return "\n".join(lines)


def _parse_edge_line(line: str) -> tuple[str, str] | None:
    """Try to parse a single edge from a line.

    Supports: "0 1", "0, 1", "(0, 1)", "0->1", "0 -> 1"
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # Remove parentheses
    line = line.strip("()")

    # Try arrow format: "0 -> 1" or "0->1"
    if "->" in line:
        parts = line.split("->")
        if len(parts) == 2:
            u = parts[0].strip().strip("'\"")
            v = parts[1].strip().strip("'\"")
            if u and v:
                return (u, v)

    # Try comma: "0, 1"
    if "," in line:
        parts = [p.strip().strip("'\"") for p in line.split(",")]
        if len(parts) == 2 and parts[0] and parts[1]:
            return (parts[0], parts[1])

    # Try space: "0 1"
    parts = line.split()
    if len(parts) == 2:
        u = parts[0].strip("'\"")
        v = parts[1].strip("'\"")
        if u and v:
            return (u, v)

    return None


def _detect_directed(lines: list[str]) -> bool:
    """Detect if the graph is directed from header lines."""
    for line in lines:
        lower = line.lower().strip()
        if lower.startswith("directed"):
            return "true" in lower
    return False
