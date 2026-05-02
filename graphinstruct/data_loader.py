"""Load and validate instruction datasets from data/instructions/."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# Default data directory relative to project root
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "instructions"


@dataclass(frozen=True)
class Instruction:
    """A single benchmark instruction."""

    id: str
    level: int
    instruction: str
    explicit_constraints: tuple[str, ...]
    implicit_constraints: tuple[str, ...]
    graph_sizes: tuple[str, ...]
    feasible: bool
    reference_solutions: tuple[str, ...] = ()
    base_graph: Optional[str] = None  # Code-style format for L5 base graph
    graph_edits: tuple[str, ...] = ()  # GraphEdit format reference edits for L5
    reasoning_steps: tuple[dict, ...] = ()  # GRAPHIA micro-eval step decomposition
    domain: Optional[str] = None  # L4 domain label (social/citation/biological/...)

    @classmethod
    def from_dict(cls, d: dict) -> "Instruction":
        return cls(
            id=d["id"],
            level=d["level"],
            instruction=d["instruction"],
            explicit_constraints=tuple(d["explicit_constraints"]),
            implicit_constraints=tuple(d["implicit_constraints"]),
            graph_sizes=tuple(d["graph_sizes"]),
            feasible=d.get("feasible", True),
            reference_solutions=tuple(d.get("reference_solutions", [])),
            base_graph=d.get("base_graph"),
            graph_edits=tuple(d.get("graph_edits", [])),
            reasoning_steps=tuple(d.get("reasoning_steps", [])),
            domain=d.get("domain"),
        )


def load_level(level: int, data_dir: Optional[Path] = None) -> list[Instruction]:
    """Load all instructions for a given level.

    Args:
        level: Instruction level (0-5).
        data_dir: Directory containing level_X.json files.

    Returns:
        List of Instruction objects.

    Raises:
        FileNotFoundError: If the level file doesn't exist.
    """
    data_dir = data_dir or _DATA_DIR
    path = data_dir / f"level_{level}.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [Instruction.from_dict(d) for d in raw]


def load_all_levels(
    levels: Optional[Sequence[int]] = None,
    data_dir: Optional[Path] = None,
) -> dict[int, list[Instruction]]:
    """Load instructions for multiple levels.

    Args:
        levels: Which levels to load. Defaults to [0, 1, 2, 3].
        data_dir: Directory containing level_X.json files.

    Returns:
        Dict mapping level -> list of Instructions.
    """
    if levels is None:
        levels = [0, 1, 2, 3, 4, 5]
    return {lvl: load_level(lvl, data_dir) for lvl in levels}
