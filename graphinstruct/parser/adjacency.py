"""Parser for adjacency list graph format.

Format example:
    Nodes: 0, 1, 2, 3, 4
    Adjacency list:
    0: 1 2
    1: 0 3
    2: 0
    3: 1 4
    4: 3
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import networkx as nx

from graphinstruct.parser.code_style import ParseResult


def parse(text: str) -> ParseResult:
    """Parse an adjacency list format string into a ParseResult.

    Supports formats:
    - "0: 1 2 3" (space separated)
    - "0: 1, 2, 3" (comma separated)
    - Optional "Nodes:" header line
    - Optional "Adjacency list:" header line

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

    # Extract node list if explicit header exists
    node_list: list[str] = []
    adj_lines: list[str] = []
    in_adj_section = False

    for line in lines:
        lower = line.lower()
        # Skip header/comment lines
        if lower.startswith("adjacency") or lower.startswith("adj"):
            in_adj_section = True
            continue
        if lower.startswith("nodes:"):
            # Parse explicit node list: "Nodes: 0, 1, 2, 3"
            node_part = line.split(":", 1)[1].strip()
            node_list = [
                n.strip().strip("'\"") for n in node_part.split(",") if n.strip()
            ]
            continue
        if lower.startswith("directed"):
            continue
        # Lines with ":" are adjacency entries
        if ":" in line:
            in_adj_section = True
            adj_lines.append(line)
        elif in_adj_section:
            # Continuation or malformed — skip
            continue

    if not adj_lines:
        raise ValueError("No adjacency list entries found")

    # Parse adjacency entries
    G = graph_cls()
    if node_list:
        G.add_nodes_from(node_list)

    for line in adj_lines:
        parts = line.split(":", 1)
        source = parts[0].strip().strip("'\"")
        if not source:
            continue
        G.add_node(source)

        neighbors_str = parts[1].strip() if len(parts) > 1 else ""
        if not neighbors_str:
            continue

        # Split by comma or space
        if "," in neighbors_str:
            neighbors = [
                n.strip().strip("'\"") for n in neighbors_str.split(",") if n.strip()
            ]
        else:
            neighbors = [
                n.strip().strip("'\"") for n in neighbors_str.split() if n.strip()
            ]

        for nb in neighbors:
            G.add_node(nb)
            G.add_edge(source, nb)

    return ParseResult(
        parse_type="Graph",
        metadata={"format": "adjacency"},
        graph=G,
    )


def serialize(
    graph: nx.Graph,
    name: str = "",
    **extra_metadata: Any,
) -> str:
    """Serialize a NetworkX graph to adjacency list format.

    Args:
        graph: The NetworkX graph to serialize.
        name: Optional graph name (included as comment).

    Returns:
        Adjacency list format string.
    """
    lines: list[str] = []
    is_directed = isinstance(graph, nx.DiGraph)

    # Header
    if name:
        lines.append(f"# {name}")
    if is_directed:
        lines.append("Directed: true")

    # Node list
    nodes = sorted(graph.nodes(), key=str)
    lines.append(f"Nodes: {', '.join(str(n) for n in nodes)}")
    lines.append("Adjacency list:")

    # Adjacency entries
    for node in nodes:
        neighbors = sorted(graph.neighbors(node))
        if neighbors:
            lines.append(f"{node}: {' '.join(str(n) for n in neighbors)}")
        else:
            lines.append(f"{node}:")

    return "\n".join(lines)


def _detect_directed(lines: list[str]) -> bool:
    """Detect if the graph is directed from header lines."""
    for line in lines:
        lower = line.lower().strip()
        if lower.startswith("directed"):
            return "true" in lower
    return False
