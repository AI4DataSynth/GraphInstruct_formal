"""Parser for InstructGraph code-style graph format.

Handles parsing and serialization of the code-style format:
    Graph[name='task-001', type='tree', nodes=10] {
        node_list = ['n0', 'n1', ...];
        edge_list = [('n0','n1'), ...];
    }
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class ParseResult:
    """Result of parsing a code-style graph format string."""

    parse_type: str  # "Graph" or "GraphEdit"
    metadata: dict[str, Any]
    graph: nx.Graph
    # GraphEdit specific
    add_edges: tuple[tuple[str, str], ...] = ()
    remove_edges: tuple[tuple[str, str], ...] = ()
    reasoning: str = ""


def parse(text: str) -> ParseResult:
    """Parse a code-style graph format string into a ParseResult.

    Args:
        text: The code-style graph format string.

    Returns:
        ParseResult with the parsed graph and metadata.

    Raises:
        ValueError: If the format is invalid.
    """
    text = text.strip()

    # Strip markdown code-block fences (```plaintext, ```python3, ```c++, ```, etc.)
    text = re.sub(r"```\S*\s*\n?", "", text)

    # Normalize DiGraph prefix to Graph (directed is inferred from header attributes)
    text = re.sub(r"\bDiGraph\b", "Graph", text)

    # Extract header — try match (start of string) first, then search (anywhere)
    header_match = re.match(r"(Graph(?:Edit)?)\s*\[([^\]]*)\]", text)
    if not header_match:
        # Fallback: find Graph[...]{...} anywhere in text (CoT outputs)
        header_match = re.search(r"(Graph(?:Edit)?)\s*\[([^\]]*)\]", text)
        if header_match:
            # Trim text to start from the Graph[...] block
            text = text[header_match.start() :]
            header_match = re.match(r"(Graph(?:Edit)?)\s*\[([^\]]*)\]", text)
    if not header_match:
        raise ValueError(
            "Invalid format: expected 'Graph[...]' or 'GraphEdit[...]' header"
        )

    parse_type = header_match.group(1)
    header_content = header_match.group(2)
    metadata = _parse_header(header_content)

    # Extract body between { and } (search from after header to avoid header braces)
    body_start = text.find("{", header_match.end())
    body_end = text.rfind("}")
    if body_start == -1 or body_end == -1 or body_end <= body_start:
        raise ValueError("Invalid format: expected body between { }")

    body = text[body_start + 1 : body_end]
    statements = _parse_body_statements(body)

    is_directed = metadata.get("directed", False)
    graph_cls = nx.DiGraph if is_directed else nx.Graph

    if parse_type == "Graph":
        return _build_graph_result(statements, metadata, graph_cls)
    else:
        return _build_graph_edit_result(statements, metadata, graph_cls)


def serialize(
    graph: nx.Graph,
    name: str = "",
    graph_type: str = "",
    **extra_metadata: Any,
) -> str:
    """Serialize a NetworkX graph to code-style format.

    Args:
        graph: The NetworkX graph to serialize.
        name: Graph name for the header.
        graph_type: Graph type for the header.
        **extra_metadata: Additional key-value pairs for the header.

    Returns:
        Code-style format string.
    """
    # Build header
    header_parts: list[str] = []
    if name:
        header_parts.append(f"name='{name}'")
    if graph_type:
        header_parts.append(f"type='{graph_type}'")
    if isinstance(graph, nx.DiGraph):
        header_parts.append("directed=true")
    header_parts.append(f"nodes={graph.number_of_nodes()}")
    for key, value in extra_metadata.items():
        header_parts.append(f"{key}={_serialize_value(value)}")

    header = f"Graph[{', '.join(header_parts)}]"

    # Build body
    lines: list[str] = []

    # Node list
    nodes = sorted(graph.nodes(), key=str)
    node_strs = [f"'{n}'" for n in nodes]
    node_line = ", ".join(node_strs)
    if len(node_line) > 60:
        formatted = _format_long_list(node_strs, indent=8)
        lines.append(f"    node_list = [{formatted}];")
    else:
        lines.append(f"    node_list = [{', '.join(node_strs)}];")

    # Edge list
    edges = list(graph.edges())
    if edges:
        edge_strs = [f"('{u}','{v}')" for u, v in edges]
        edge_line = ", ".join(edge_strs)
        if len(edge_line) > 60:
            # Multi-line formatting for long edge lists
            formatted = _format_long_list(edge_strs, indent=8)
            lines.append(f"    edge_list = [{formatted}];")
        else:
            lines.append(f"    edge_list = [{edge_line}];")
    else:
        lines.append("    edge_list = [];")

    # Node attributes
    node_attr_lines: list[str] = []
    for node in nodes:
        attrs = graph.nodes[node]
        for attr_name, attr_value in sorted(attrs.items()):
            val_str = _serialize_value(attr_value)
            node_attr_lines.append(f"    {node}.{attr_name} = {val_str};")
    if node_attr_lines:
        lines.append("")
        lines.extend(node_attr_lines)

    # Edge attributes (skip internal metadata)
    # Optimization: if all edges share the same value for an attribute,
    # emit it once as a graph-level property (e.g., edge_type = 'friends')
    # instead of per-edge (saves massive token count for large graphs).
    edge_attr_keys: set[str] = set()
    for _, _, data in graph.edges(data=True):
        for k in data:
            if not k.startswith("_"):
                edge_attr_keys.add(k)

    uniform_edge_attrs: dict[str, Any] = {}
    per_edge_attrs: dict[tuple[str, str], dict[str, Any]] = {}

    for attr_key in sorted(edge_attr_keys):
        values = set()
        for _, _, data in graph.edges(data=True):
            if attr_key in data:
                values.add(data[attr_key])
        if len(values) == 1:
            # All edges have the same value — promote to graph-level
            uniform_edge_attrs[f"edge_{attr_key}"] = values.pop()
        else:
            # Different values — keep per-edge
            for u, v, data in graph.edges(data=True):
                if attr_key in data:
                    key = (u, v)
                    if key not in per_edge_attrs:
                        per_edge_attrs[key] = {}
                    per_edge_attrs[key][attr_key] = data[attr_key]

    edge_attr_lines: list[str] = []
    for (u, v), attrs in sorted(per_edge_attrs.items(), key=lambda x: str(x[0])):
        for attr_name, attr_value in sorted(attrs.items()):
            val_str = _serialize_value(attr_value)
            edge_attr_lines.append(f"    edge({u},{v}).{attr_name} = {val_str};")
    if edge_attr_lines:
        lines.append("")
        lines.extend(edge_attr_lines)

    # Graph-level properties (skip internal metadata)
    graph_props = {k: v for k, v in graph.graph.items() if not k.startswith("_")}
    graph_props.update(uniform_edge_attrs)  # Add promoted edge attributes
    if graph_props:
        lines.append("")
        for key, value in sorted(graph_props.items()):
            lines.append(f"    {key} = {_serialize_value(value)};")

    body = "\n".join(lines)
    return f"{header} {{\n{body}\n}}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_header(content: str) -> dict[str, Any]:
    """Parse header key=value pairs like: name='task-001', type='tree', nodes=10"""
    result: dict[str, Any] = {}
    # Match key=value pairs where value is quoted string or number/bool
    pattern = r"""(\w+)\s*=\s*(?:'([^']*)'|"([^"]*)"|(\w+(?:\.\d+)?))"""
    for match in re.finditer(pattern, content):
        key = match.group(1)
        if match.group(2) is not None:
            result[key] = match.group(2)
        elif match.group(3) is not None:
            result[key] = match.group(3)
        else:
            raw = match.group(4)
            result[key] = _coerce_scalar(raw)
    return result


def _parse_body_statements(body: str) -> list[str]:
    """Split body into individual statements, handling multi-line lists."""
    # Remove comment lines
    lines = body.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Split by semicolons, but respect brackets and quotes
    statements: list[str] = []
    current: list[str] = []
    bracket_depth = 0
    in_quote: str | None = None

    for char in text:
        if in_quote:
            current.append(char)
            if char == in_quote:
                in_quote = None
        elif char in ('"', "'"):
            current.append(char)
            in_quote = char
        elif char == "[":
            bracket_depth += 1
            current.append(char)
        elif char == "]":
            bracket_depth -= 1
            current.append(char)
        elif char == ";" and bracket_depth == 0:
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
        else:
            current.append(char)

    # Handle trailing content without semicolon
    leftover = "".join(current).strip()
    if leftover:
        statements.append(leftover)

    return statements


def _parse_assignment(stmt: str) -> tuple[str, Any]:
    """Parse 'lhs = rhs' from a statement. Returns (lhs, parsed_value)."""
    eq_pos = stmt.find("=")
    if eq_pos == -1:
        raise ValueError(f"Invalid statement (no '='): {stmt!r}")

    # Handle node.attr = value (the dot comes before =)
    lhs = stmt[:eq_pos].strip()
    rhs = stmt[eq_pos + 1 :].strip()

    value = _parse_value(rhs)
    return lhs, value


def _parse_value(raw: str) -> Any:
    """Parse a value string (list, string, number, bool)."""
    raw = raw.strip()
    if not raw:
        return None

    # Handle implicit string concatenation: "foo" "bar" -> "foobar"
    if raw.startswith(('"', "'")) and not raw.startswith(("[", "(")):
        return _parse_concatenated_string(raw)

    # Try ast.literal_eval for lists, tuples, numbers, strings
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        pass

    # Fallback: treat as bare identifier/scalar
    return _coerce_scalar(raw)


def _parse_concatenated_string(raw: str) -> str:
    """Parse possibly concatenated strings like: "hello " "world" """
    parts: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] in ('"', "'"):
            quote = raw[i]
            end = raw.find(quote, i + 1)
            if end == -1:
                # Unmatched quote: take rest of string as content
                parts.append(raw[i + 1 :])
                break
            parts.append(raw[i + 1 : end])
            i = end + 1
        else:
            i += 1
    return "".join(parts)


def _coerce_scalar(raw: str) -> int | float | bool | str:
    """Convert a raw string to int, float, bool, or leave as string."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _build_graph_result(
    statements: list[str],
    metadata: dict[str, Any],
    graph_cls: type,
) -> ParseResult:
    """Build a ParseResult for a Graph block."""
    node_list: list[str] = []
    edge_list: list[tuple[str, ...]] = []
    graph_props: dict[str, Any] = {}
    node_attrs: dict[str, dict[str, Any]] = {}
    edge_attrs: dict[tuple[str, str], dict[str, Any]] = {}

    for stmt in statements:
        lhs, value = _parse_assignment(stmt)

        if lhs == "node_list":
            node_list = value if isinstance(value, list) else list(value)
        elif lhs == "edge_list":
            edge_list = value if isinstance(value, list) else list(value)
        elif lhs.startswith("edge("):
            # Edge attribute: edge(u,v).attr = val
            import re

            m = re.match(r"edge\(([^,]+),([^)]+)\)\.(.+)", lhs)
            if m:
                u, v, attr_name = (
                    m.group(1).strip(),
                    m.group(2).strip(),
                    m.group(3).strip(),
                )
                key = (u, v)
                if key not in edge_attrs:
                    edge_attrs[key] = {}
                edge_attrs[key][attr_name] = value
        elif "." in lhs:
            # Node attribute: n0.community = 0
            node_name, attr_name = lhs.split(".", 1)
            node_name = node_name.strip()
            attr_name = attr_name.strip()
            if node_name not in node_attrs:
                node_attrs[node_name] = {}
            node_attrs[node_name][attr_name] = value
        else:
            graph_props[lhs] = value

    # Build graph
    G = graph_cls()
    G.add_nodes_from(node_list)
    for edge in edge_list:
        if len(edge) == 2:
            G.add_edge(edge[0], edge[1])
        elif len(edge) >= 3:
            G.add_edge(edge[0], edge[1], weight=edge[2])

    # Attach node attributes
    for node_name, attrs in node_attrs.items():
        if node_name in G.nodes:
            for attr_name, attr_value in attrs.items():
                G.nodes[node_name][attr_name] = attr_value

    # Attach edge attributes
    for (u, v), attrs in edge_attrs.items():
        if G.has_edge(u, v):
            for attr_name, attr_value in attrs.items():
                G[u][v][attr_name] = attr_value

    # Attach graph properties
    for key, value in graph_props.items():
        G.graph[key] = value

    return ParseResult(
        parse_type="Graph",
        metadata=metadata,
        graph=G,
    )


def _build_graph_edit_result(
    statements: list[str],
    metadata: dict[str, Any],
    graph_cls: type,
) -> ParseResult:
    """Build a ParseResult for a GraphEdit block."""
    add_edges: list[tuple[str, str]] = []
    remove_edges: list[tuple[str, str]] = []
    reasoning = ""
    graph_props: dict[str, Any] = {}

    for stmt in statements:
        lhs, value = _parse_assignment(stmt)

        if lhs == "add_edges":
            add_edges = [tuple(e) for e in (value or [])]
        elif lhs == "remove_edges":
            remove_edges = [tuple(e) for e in (value or [])]
        elif lhs == "reasoning":
            reasoning = str(value)
        else:
            graph_props[lhs] = value

    G = graph_cls()
    for key, value in graph_props.items():
        G.graph[key] = value

    return ParseResult(
        parse_type="GraphEdit",
        metadata=metadata,
        graph=G,
        add_edges=tuple(add_edges),
        remove_edges=tuple(remove_edges),
        reasoning=reasoning,
    )


def _serialize_value(value: Any) -> str:
    """Serialize a Python value to code-style format string."""
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(value)


def _format_long_list(items: list[str], indent: int = 8) -> str:
    """Format a long list across multiple lines."""
    prefix = " " * indent
    lines: list[str] = []
    current_line: list[str] = []
    current_len = 0

    for item in items:
        if current_len + len(item) + 2 > 60 and current_line:
            lines.append(", ".join(current_line) + ",")
            current_line = [item]
            current_len = len(item)
        else:
            current_line.append(item)
            current_len += len(item) + 2

    if current_line:
        lines.append(", ".join(current_line))

    if len(lines) == 1:
        return lines[0]

    joined = ("\n" + prefix).join(lines)
    return f"\n{prefix}{joined}"


# ---------------------------------------------------------------------------
# Directed edge recovery (Phase 0b fix)
# ---------------------------------------------------------------------------


def recover_directed_edges(G: nx.Graph, raw_output: str) -> tuple[nx.DiGraph, bool]:
    """Recover directed edges from raw_output when parser lost direction info.

    When a model writes edge_list = [('a','b'), ('c','d')] but omits
    directed=true from the Graph header, the parser creates an undirected
    graph. This function re-parses the raw edge_list to restore direction.

    Caller is responsible for checking that:
    - The instruction requires directed=true
    - The graph is currently undirected

    Args:
        G: Undirected graph whose edges should be directed.
        raw_output: Raw LLM output containing the edge_list.

    Returns:
        (digraph, recovered_from_raw): The directed graph, and True if
        edges were recovered from raw_output (False = simple fallback).
    """
    # Try to extract original edge_list from raw_output
    edge_match = re.search(r"edge_list\s*=\s*\[(.*?)\]", raw_output, re.DOTALL)
    if edge_match:
        tuple_pattern = re.compile(r"\(\s*['\"](.+?)['\"]\s*,\s*['\"](.+?)['\"]\s*\)")
        directed_edges = tuple_pattern.findall(edge_match.group(1))
        if directed_edges:
            original_nodes = set(G.nodes())
            # Filter: only keep edges whose endpoints are in the original graph.
            # Raw edge_list may have different names (casing, formatting) that
            # would implicitly create spurious nodes via add_edge.
            valid_edges = [
                (u, v)
                for u, v in directed_edges
                if u in original_nodes and v in original_nodes
            ]
            if valid_edges:
                DG = nx.DiGraph()
                DG.add_nodes_from(G.nodes(data=True))
                for u, v in valid_edges:
                    edge_data = G.edges.get((u, v), {})
                    DG.add_edge(u, v, **edge_data)
                DG.graph.update(G.graph)
                return DG, True

    # Fallback: convert using current edge order (arbitrary direction)
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        DG.add_edge(u, v, **data)
    DG.graph.update(G.graph)
    return DG, False
