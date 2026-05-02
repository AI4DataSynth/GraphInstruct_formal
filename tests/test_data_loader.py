"""Tests for graphinstruct.data_loader."""

import unittest
from pathlib import Path

from graphinstruct.data_loader import Instruction, load_all_levels, load_level

# Use the actual data directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "instructions"


class TestInstruction(unittest.TestCase):
    def test_from_dict(self):
        d = {
            "id": "L0-001",
            "level": 0,
            "instruction": "Generate a graph with 5 nodes.",
            "explicit_constraints": ["num_nodes=5"],
            "implicit_constraints": [],
            "graph_sizes": ["small"],
            "reference_solutions": [],
            "feasible": True,
        }
        inst = Instruction.from_dict(d)
        self.assertEqual(inst.id, "L0-001")
        self.assertEqual(inst.level, 0)
        self.assertEqual(inst.explicit_constraints, ("num_nodes=5",))
        self.assertTrue(inst.feasible)

    def test_frozen(self):
        inst = Instruction(
            id="L0-001",
            level=0,
            instruction="test",
            explicit_constraints=(),
            implicit_constraints=(),
            graph_sizes=(),
            feasible=True,
        )
        with self.assertRaises(AttributeError):
            inst.id = "changed"


class TestLoadLevel(unittest.TestCase):
    def test_load_level_0(self):
        instructions = load_level(0, DATA_DIR)
        self.assertGreater(len(instructions), 0)
        for inst in instructions:
            self.assertEqual(inst.level, 0)
            self.assertTrue(inst.id.startswith("L0-"))

    def test_load_level_1(self):
        instructions = load_level(1, DATA_DIR)
        self.assertGreater(len(instructions), 0)
        for inst in instructions:
            self.assertEqual(inst.level, 1)

    def test_load_nonexistent_level(self):
        with self.assertRaises(FileNotFoundError):
            load_level(99, DATA_DIR)


class TestLoadAllLevels(unittest.TestCase):
    def test_load_all_defaults(self):
        all_data = load_all_levels(data_dir=DATA_DIR)
        for lvl in range(6):
            self.assertIn(lvl, all_data)

    def test_load_specific_levels(self):
        data = load_all_levels(levels=[0, 2], data_dir=DATA_DIR)
        self.assertIn(0, data)
        self.assertIn(2, data)
        self.assertNotIn(1, data)

    def test_instruction_counts(self):
        all_data = load_all_levels(data_dir=DATA_DIR)
        self.assertEqual(len(all_data[0]), 100)
        self.assertEqual(len(all_data[1]), 200)
        self.assertEqual(len(all_data[2]), 200)
        self.assertEqual(len(all_data[3]), 150)
        self.assertEqual(len(all_data[4]), 100)
        self.assertEqual(len(all_data[5]), 50)


if __name__ == "__main__":
    unittest.main()
