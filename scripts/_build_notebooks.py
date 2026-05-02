"""Generate the two example Jupyter notebooks deterministically.

Run from repo root:
    python scripts/_build_notebooks.py

Produces:
    examples/01_quickstart.ipynb
    examples/02_eval_your_model.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
OUT = WORK / "examples"
OUT.mkdir(parents=True, exist_ok=True)


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": list(lines),
    }


def write_nb(name: str, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    fp = OUT / name
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  -> {fp.relative_to(WORK)}")


# ============================================================
# 01 Quickstart — 5-min hands-on tour
# ============================================================
nb1 = [
    md(
        "# GraphInstruct Quickstart (5 minutes)\n",
        "\n",
        "This notebook walks you through:\n",
        "1. Loading the 800-instruction benchmark\n",
        "2. Inspecting the 6 complexity levels (L0–L5)\n",
        "3. Running the D1, D4, D5 evaluation pipeline on a small mock model\n",
        "4. Reading the resulting quality scores\n",
        "\n",
        "**No API key needed.** This notebook uses a deterministic mock generator that\n",
        "always emits the first reference solution — useful for verifying the eval\n",
        "pipeline end-to-end without spending tokens.\n",
    ),
    md(
        "## Setup\n",
        "\n",
        "If you haven't installed yet:\n",
        "```bash\n",
        "pip install -e ..   # from the repo root\n",
        "```\n",
        "\n",
        "Windows users: set these env vars **before** launching Jupyter:\n",
        "```bash\n",
        "export KMP_DUPLICATE_LIB_OK=TRUE\n",
        "export PYTHONIOENCODING=utf-8\n",
        "```\n",
    ),
    code(
        "import os\n",
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "# Allow running this notebook either from examples/ or from repo root\n",
        "REPO = Path.cwd()\n",
        "if (REPO.name == 'examples'):\n",
        "    REPO = REPO.parent\n",
        "sys.path.insert(0, str(REPO))\n",
        "os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')\n",
        "os.environ.setdefault('PYTHONIOENCODING', 'utf-8')\n",
        "print(f'Repo root: {REPO}')\n",
    ),
    md(
        "## 1. Load the 800-instruction benchmark\n",
        "\n",
        "Each instruction is a (natural-language query, constraint specification, reference solutions) tuple.\n",
    ),
    code(
        "from graphinstruct.data_loader import load_all_levels\n",
        "\n",
        "instructions = load_all_levels(data_dir=REPO / 'data' / 'instructions')\n",
        "for L in sorted(instructions):\n",
        "    print(f'  L{L}: {len(instructions[L])} instructions')\n",
        "total = sum(len(v) for v in instructions.values())\n",
        "print(f'\\nTotal: {total} instructions across 6 levels')\n",
    ),
    md(
        "## 2. Inspect a sample instruction\n",
        "\n",
        "Let's look at one L1 instruction (single explicit constraint).\n",
    ),
    code(
        "sample = instructions[1][0]\n",
        "print(f'ID: {sample.id}')\n",
        "print(f'Level: L{sample.level}')\n",
        "print(f'Instruction: {sample.instruction}')\n",
        "print(f'Explicit constraints: {sample.explicit_constraints}')\n",
        "print(f'Implicit constraints: {sample.implicit_constraints}')\n",
        "print(f'Number of reference solutions: {len(sample.reference_solutions)}')\n",
    ),
    md(
        "## 3. Run a deterministic mock generator\n",
        "\n",
        "We'll use a mock LLM that just echoes the first reference solution. This\n",
        "lets us verify the evaluation pipeline gives sensible numbers (it should\n",
        "score near-perfect on D1 and D4 since outputs match the references).\n",
    ),
    code(
        "# Pick 5 instructions from each of L0/L1/L2/L3 (skip L4/L5 for speed)\n",
        "mini = []\n",
        "for L in (0, 1, 2, 3):\n",
        "    for inst in instructions[L][:5]:\n",
        "        if inst.feasible and inst.reference_solutions:\n",
        "            mini.append(inst)\n",
        "print(f'Mini set: {len(mini)} instructions')\n",
    ),
    code(
        "# Build mock outputs (echo first reference)\n",
        "mock_outputs = []\n",
        "for inst in mini:\n",
        "    mock_outputs.append({\n",
        "        'instruction_id': inst.id,\n",
        "        'level': inst.level,\n",
        "        'graph_serialized': inst.reference_solutions[0],\n",
        "    })\n",
        "print(f'Generated {len(mock_outputs)} mock outputs')\n",
    ),
    md(
        "## 4. Score with D1 (structural) and D4 (instruction match)\n",
        "\n",
        "These are the two cheapest metrics — no GPU, no BERT, no API.\n",
    ),
    code(
        "from graphinstruct.parser import parse\n",
        "from graphinstruct.metrics.structural import valid_rate\n",
        "from graphinstruct.metrics.instruction import instruction_score\n",
        "\n",
        "import statistics\n",
        "\n",
        "d1_per_inst = []\n",
        "d4_per_inst = []\n",
        "for inst, out in zip(mini, mock_outputs):\n",
        "    try:\n",
        "        result = parse(out['graph_serialized'])\n",
        "        g = result.graph\n",
        "        d1 = valid_rate([g], constraints=list(inst.explicit_constraints))\n",
        "        d4 = instruction_score(g, list(inst.explicit_constraints))\n",
        "    except Exception:\n",
        "        d1, d4 = 0.0, 0.0\n",
        "    d1_per_inst.append(d1)\n",
        "    d4_per_inst.append(d4)\n",
        "\n",
        "print(f'D1 (Valid Rate)         mean: {statistics.mean(d1_per_inst):.3f}')\n",
        "print(f'D4 (Instruction Match)  mean: {statistics.mean(d4_per_inst):.3f}')\n",
        "print('\\nSince outputs == references, both should score near 1.0.')\n",
    ),
    md(
        "## 5. What's next?\n",
        "\n",
        "- **Try a real model**: see [`02_eval_your_model.ipynb`](02_eval_your_model.ipynb)\n",
        "  for the recipe to plug in your own LLM\n",
        "- **Reproduce paper numbers**: see [`docs/REPRODUCE.md`](../docs/REPRODUCE.md)\n",
        "- **Inspect cached results**: open `results/quality/<model>-<strategy>.quality.json`\n",
        "  for any of the 45 (model, strategy) cells in the paper\n",
        "- **Run the D5 robustness ablation** (Appendix C, Tab. 3):\n",
        "  ```bash\n",
        "  python scripts/d5_robustness.py\n",
        "  ```\n",
    ),
]
write_nb("01_quickstart.ipynb", nb1)


# ============================================================
# 02 Evaluate your own model
# ============================================================
nb2 = [
    md(
        "# Evaluate your own model with GraphInstruct\n",
        "\n",
        "This notebook shows how to plug **any LLM** (commercial API or local model)\n",
        "into the GraphInstruct evaluation pipeline. You'll get back per-level\n",
        "scores comparable to the 45 baseline cells in [`results/quality/`](../results/quality/).\n",
        "\n",
        "## What you'll need\n",
        "\n",
        "- A way to generate text from your model (any callable that takes a string\n",
        "  and returns a string)\n",
        "- ~1 minute per (instruction, sample) pair on most APIs\n",
        "- Optional: GPU for D2 (G-BERTScore) and D3 (Node Classification Gap)\n",
        "  metrics; CPU works for D1 / D4 / D5\n",
    ),
    code(
        "import os\n",
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "REPO = Path.cwd()\n",
        "if REPO.name == 'examples':\n",
        "    REPO = REPO.parent\n",
        "sys.path.insert(0, str(REPO))\n",
        "os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')\n",
        "os.environ.setdefault('PYTHONIOENCODING', 'utf-8')\n",
    ),
    md(
        "## Step 1: Define your model wrapper\n",
        "\n",
        "GraphInstruct's `run_baseline.py` already supports OpenAI-compatible APIs\n",
        "(`--provider openai --api-base <URL>`), Anthropic, Alibaba Bailian, and\n",
        "DeepSeek. If you're plugging in a custom model, write a single function:\n",
        "\n",
        "```python\n",
        "def my_model(prompt: str, **kwargs) -> str:\n",
        '    "Return the LLM\'s text output."\n',
        "    ...\n",
        "```\n",
        "\n",
        "Below is a minimal stub — replace with your real model.\n",
    ),
    code(
        "def my_model(prompt: str) -> str:\n",
        '    """Replace this with your real LLM call."""\n',
        "    # Example: a constant trivial-tree generator (intentionally weak baseline)\n",
        "    return (\n",
        "        \"Graph[name='trivial', nodes=3] {\\n\"\n",
        '        "    node_list = [0, 1, 2];\\n"\n',
        '        "    edge_list = [(0, 1), (1, 2)];\\n"\n',
        '        "}"\n',
        "    )\n",
    ),
    md(
        "## Step 2: Pick a small slice for fast iteration\n",
        "\n",
        "When wiring up a new model, run on a 10-instruction subset first to\n",
        "verify everything works before committing to the full 800.\n",
    ),
    code(
        "from graphinstruct.data_loader import load_all_levels\n",
        "\n",
        "instructions = load_all_levels(data_dir=REPO / 'data' / 'instructions')\n",
        "\n",
        "# Take 2 from each of L0/L1/L2/L3, 1 from L4/L5 = 10 total\n",
        "counts = {0: 2, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1}\n",
        "mini = [\n",
        "    inst\n",
        "    for L, n in counts.items()\n",
        "    for inst in [i for i in instructions[L] if i.feasible][:n]\n",
        "]\n",
        "print(f'Mini set: {len(mini)} instructions across L0–L5')\n",
    ),
    md(
        "## Step 3: Generate outputs\n",
        "\n",
        "Loop over the mini set and call your model. With the trivial stub above\n",
        "this runs in <1 s; with a real API it'll take ~1 minute.\n",
    ),
    code(
        "from tqdm import tqdm\n",
        "\n",
        "outputs = []\n",
        "for inst in tqdm(mini, desc='Generating'):\n",
        "    text = my_model(inst.instruction)\n",
        "    outputs.append({\n",
        "        'instruction_id': inst.id,\n",
        "        'level': inst.level,\n",
        "        'graph_serialized': text,\n",
        "    })\n",
        "print(f'\\n{len(outputs)} outputs collected')\n",
    ),
    md(
        "## Step 4: Score with D1 + D4 (structural + instruction match)\n",
        "\n",
        "These are the two highest-weight dimensions across all 6 levels and\n",
        "require no GPU.\n",
    ),
    code(
        "from collections import defaultdict\n",
        "import statistics\n",
        "\n",
        "from graphinstruct.parser import parse\n",
        "from graphinstruct.metrics.structural import valid_rate\n",
        "from graphinstruct.metrics.instruction import instruction_score\n",
        "\n",
        "per_level = defaultdict(list)\n",
        "for inst, out in zip(mini, outputs):\n",
        "    try:\n",
        "        result = parse(out['graph_serialized'])\n",
        "        g = result.graph\n",
        "        d1 = valid_rate([g], constraints=list(inst.explicit_constraints))\n",
        "        d4 = instruction_score(g, list(inst.explicit_constraints))\n",
        "    except Exception:\n",
        "        d1, d4 = 0.0, 0.0\n",
        "    per_level[inst.level].append({'D1': d1, 'D4': d4})\n",
        "\n",
        "for L in sorted(per_level):\n",
        "    d1_mean = statistics.mean(r['D1'] for r in per_level[L])\n",
        "    d4_mean = statistics.mean(r['D4'] for r in per_level[L])\n",
        "    print(f'L{L}: n={len(per_level[L])}  D1={d1_mean:.3f}  D4={d4_mean:.3f}')\n",
    ),
    md(
        "## Step 5: Compare against the 12 baseline LLMs\n",
        "\n",
        "Once you've evaluated on the full 800 instructions, you can drop your\n",
        "scores onto the leaderboard for direct comparison.\n",
    ),
    code(
        "import csv\n",
        "from pathlib import Path\n",
        "\n",
        "leaderboard = REPO / 'results' / 'leaderboards' / 'tab1_quality_top15.csv'\n",
        "with open(leaderboard, encoding='utf-8') as f:\n",
        "    reader = csv.DictReader(f)\n",
        "    rows = list(reader)\n",
        "print(f'\\n{leaderboard.name}:')\n",
        "for r in rows[:5]:\n",
        "    print(f\"  #{r['Rank']}  {r['Model']:<25} {r['Strategy']}  Q={r['Quality (Total)']}\")\n",
    ),
    md(
        "## Going further\n",
        "\n",
        "- **Full 800-instruction run**: drop the `counts` slice above and loop\n",
        "  over all 800 instructions; budget ~1 hr for a fast model, ~6 hrs for a\n",
        "  reasoning model\n",
        '- **All 5 dimensions**: install `[full]` (`pip install -e ".[full]"`) for\n',
        "  D2 (BERTScore-based) and D3 (lightweight GCN-based) metrics\n",
        "- **Use `scripts/run_baseline.py` directly**: handles\n",
        "  retries, rate limiting, partial-progress checkpointing, and writes the\n",
        "  same `quality.json` schema used by the 45 baseline cells\n",
    ),
]
write_nb("02_eval_your_model.ipynb", nb2)

print("\nBoth notebooks generated.")
