# Dataset Schema / 数据集结构

This directory contains the GraphInstruct benchmark data. It is organized into two pools:

```
data/
├── instructions/             # 800 instructions, the benchmark itself
│   ├── level_0.json          # 100 L0 instructions (format generation)
│   ├── level_1.json          # 200 L1 instructions (single explicit constraint)
│   ├── level_2.json          # 200 L2 instructions (multi-constraint composition)
│   ├── level_3.json          # 150 L3 instructions (numerical attribute control)
│   ├── level_4.json          # 100 L4 instructions (domain semantics)
│   └── level_5.json          # 50  L5 instructions (multi-step graph editing)
└── reference_pools/          # 4,163 reference graphs for D1 / D3 distributional metrics
    ├── l3_synthetic/         # 3,115 graphs over 15 attribute subgroups × 3 sizes
    │   ├── density-low-small.pkl
    │   ├── density-low-medium.pkl
    │   ├── ... (15 × 3 = 45 files)
    └── l4_real/              # 1,048 real-world graphs across 9 domains
        ├── citation.pkl
        ├── social.pkl
        ├── biological.pkl
        ├── infrastructure.pkl
        ├── communication.pkl
        ├── ecological.pkl
        ├── general.pkl
        └── ba-{s,m,l}.pkl    # synthetic BA controls
```

### `data/instructions/level_X.json` — instruction file schema

Each `level_X.json` is a JSON list. Each list entry is one instruction with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier, format `L{level}-{seq:03d}` (e.g., `L1-042`) |
| `level` | int | Instruction level, 0–5 |
| `instruction` | str | Natural-language instruction issued to the LLM |
| `explicit_constraints` | list[str] | Constraints stated in the instruction itself (e.g., `graph_type=tree`, `num_nodes=10`) |
| `implicit_constraints` | list[str] | Constraints implied by the explicit ones (e.g., a tree with n nodes implies acyclic, connected, n−1 edges) |
| `graph_sizes` | list[str] | Size buckets the instruction applies to: subset of `["small", "medium", "large"]` (small ≤20, medium 21–50, large >50) |
| `reference_solutions` | list[str] | 1 or 2 reference graphs encoded in InstructGraph code-style format; used by D2 (text similarity) and D4 (instruction match) |
| `feasible` | bool | `false` for the 9 deliberately infeasible L2 instructions (regular-degree constraints incompatible with node/edge counts); `true` otherwise |
| `domain` | str (L4 only) | Domain tag: `social` / `citation` / `biological` / `infrastructure` / `communication` / `ecological` / `general` |
| `base_graph` + `edits` | dict + list (L5 only) | L5 specifies a base graph and an ordered list of edit operations to apply |

### Example L1 instruction

```json
{
  "id": "L1-042",
  "level": 1,
  "instruction": "Generate a tree with 10 nodes.",
  "explicit_constraints": ["graph_type=tree", "num_nodes=10"],
  "implicit_constraints": ["num_edges=9", "acyclic=true", "connected=true"],
  "graph_sizes": ["small"],
  "reference_solutions": [
    "Graph[name='L1-042-ref1', nodes=10] {\n  node_list = [0, 1, 2, ..., 9];\n  edge_list = [(0,1), (1,2), ...];\n}",
    "Graph[name='L1-042-ref2', nodes=10] {\n  ...\n}"
  ],
  "feasible": true
}
```

### `data/reference_pools/*.pkl` — reference pool schema

Each `.pkl` file is a Python pickle of a list of NetworkX graph objects:

```python
import pickle
with open("data/reference_pools/l4_real/citation.pkl", "rb") as f:
    graphs = pickle.load(f)
# graphs : list[networkx.Graph]
print(len(graphs))         # number of reference graphs in this pool
print(graphs[0].number_of_nodes(), graphs[0].number_of_edges())
```

Each graph in the pool has been:

- **Size-normalized** (subgraphs sampled to the instruction's size bucket via BFS / random walk)
- **Deduplicated** (graph-isomorphism checked via Weisfeiler-Lehman hash)
- **Stripped of attributes** other than what D2 / D3 metrics require (node labels for D2; integer node IDs)

### Round-trip integrity

All 1,582 reference solutions in `reference_solutions` fields pass a round-trip parse → serialize → re-parse test enforced by `tests/test_parser.py`. Run `python -m unittest tests.test_parser -v` to verify on your end.

### Dataset statistics

```
Total instructions       : 800
  L0 (format)            : 100   (size: 100 small)
  L1 (explicit)          : 200   (size: 100 small,  60 medium, 40 large)
  L2 (multi-constraint)  : 200   (9 deliberately infeasible; 191 feasible)
  L3 (numerical attr.)   : 150
  L4 (domain semantics)  : 100   (8 domains; ~12-13 each)
  L5 (multi-step edit)   : 50

Total reference solutions: 1,582  (2 × 791 feasible instructions)
L3 synthetic pool        : 3,115 graphs across 45 (subgroup × size) cells
L4 real-world pool       : 1,048 graphs across 9 domains
```
