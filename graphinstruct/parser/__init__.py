"""Parsers for multiple graph formats: code-style, adjacency list, edge list."""

from graphinstruct.parser.code_style import ParseResult
from graphinstruct.parser.code_style import parse as parse_code
from graphinstruct.parser.code_style import serialize as serialize_code
from graphinstruct.parser.adjacency import parse as parse_adjacency
from graphinstruct.parser.adjacency import serialize as serialize_adjacency
from graphinstruct.parser.edge_list import parse as parse_edge_list
from graphinstruct.parser.edge_list import serialize as serialize_edge_list

# Backwards compatible defaults
parse = parse_code
serialize = serialize_code

FORMATS = ("code", "adjacency", "edge")

PARSERS = {
    "code": parse_code,
    "adjacency": parse_adjacency,
    "edge": parse_edge_list,
}

SERIALIZERS = {
    "code": serialize_code,
    "adjacency": serialize_adjacency,
    "edge": serialize_edge_list,
}


def parse_format(text: str, fmt: str = "code") -> ParseResult:
    """Parse text using the specified format parser.

    Args:
        text: Raw graph text.
        fmt: One of 'code', 'adjacency', 'edge'.

    Returns:
        ParseResult with parsed graph.
    """
    parser_fn = PARSERS.get(fmt)
    if parser_fn is None:
        raise ValueError(f"Unknown format: {fmt!r}. Choose from {FORMATS}")
    return parser_fn(text)


def serialize_format(graph, fmt: str = "code", **kwargs) -> str:
    """Serialize a graph using the specified format.

    Args:
        graph: NetworkX graph.
        fmt: One of 'code', 'adjacency', 'edge'.
        **kwargs: Additional arguments passed to the serializer.

    Returns:
        Serialized graph string.
    """
    serializer_fn = SERIALIZERS.get(fmt)
    if serializer_fn is None:
        raise ValueError(f"Unknown format: {fmt!r}. Choose from {FORMATS}")
    return serializer_fn(graph, **kwargs)


__all__ = [
    "parse",
    "serialize",
    "parse_format",
    "serialize_format",
    "ParseResult",
    "FORMATS",
    "PARSERS",
    "SERIALIZERS",
]
