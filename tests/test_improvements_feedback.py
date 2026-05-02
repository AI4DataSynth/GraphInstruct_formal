"""API contract tests for graphinstruct.improvements.feedback.

Day 1: skeleton only. Day 2 fills in implementation tests.
"""

import os
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import networkx as nx

from graphinstruct.data_loader import Instruction
from graphinstruct.improvements.feedback import generate_feedback


def _mk_inst(
    level=2,
    explicit=("num_nodes=3", "degree=2"),
    implicit=(),
    inst_id="L2-test",
    feasible=True,
    domain=None,
) -> Instruction:
    return Instruction(
        id=inst_id,
        level=level,
        instruction="test",
        explicit_constraints=tuple(explicit),
        implicit_constraints=tuple(implicit),
        graph_sizes=("small",),
        feasible=feasible,
        domain=domain,
    )


class TestGenerateFeedback(unittest.TestCase):
    def test_all_satisfied_returns_empty_string(self):
        # K3 satisfies num_nodes=3, degree=2
        G = nx.cycle_graph(3)
        out = generate_feedback(G, _mk_inst(), level="fine")
        self.assertEqual(out, "")

    def test_level_none_returns_fixed_retry_text(self):
        G = nx.Graph()
        out = generate_feedback(G, _mk_inst(), level="none")
        self.assertIn("try again", out.lower())

    def test_level_coarse_lists_constraint_keys(self):
        G = nx.Graph()
        out = generate_feedback(G, _mk_inst(), level="coarse")
        self.assertIn("degree", out.lower())

    def test_level_fine_includes_diagnostic_numbers(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])  # no edges -> degree=0 for all
        out = generate_feedback(G, _mk_inst(), level="fine")
        self.assertIn("degree", out.lower())
        # "fine" should include concrete node IDs or degree values
        self.assertTrue(any(ch.isdigit() for ch in out))

    def test_truncates_at_max_violations(self):
        # 10 isolated nodes with num_nodes=10 satisfied but 10 degree-3
        # implicit constraints are NOT a realistic case; build one with
        # many explicit constraints to exercise truncation.
        G = nx.Graph()
        explicit = tuple(f"num_nodes={i}" for i in range(1, 15))  # 14 constraints
        inst = _mk_inst(explicit=explicit, implicit=())
        out = generate_feedback(G, inst, level="fine", max_violations=3)
        self.assertIn("more", out.lower())

    def test_parse_failure_handled(self):
        # graph=None should not crash
        out = generate_feedback(None, _mk_inst(), level="fine")
        self.assertTrue(out)  # non-empty feedback


if __name__ == "__main__":
    unittest.main()
