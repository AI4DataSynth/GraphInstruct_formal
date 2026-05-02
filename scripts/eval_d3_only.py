"""Evaluate D3 sub-metrics only from existing JSONL checkpoints.

Computes grassmann_coherence, node_clf_gap, emb_mmd per instruction
for L3+ without re-running the full evaluation pipeline.

Usage:
    # Single file
    python scripts/eval_d3_only.py results/deepseekaiDeepSeekV3-zero-shot.jsonl

    # All JSONL files in results/
    python scripts/eval_d3_only.py results/*.jsonl

    # With PyG backend (if available)
    python scripts/eval_d3_only.py results/deepseekaiDeepSeekV3-zero-shot.jsonl

Output: results/<stem>.d3.json per input file.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from graphinstruct.data_loader import load_all_levels
from graphinstruct.metrics.embedding import (
    _HAS_NUMPY,
    _HAS_PYG,
    _HAS_TORCH,
    d3_score_batch,
    grassmann_coherence,
    node_clf_gap_batch,
    embedding_mmd,
)
from graphinstruct.parser.code_style import parse
from scripts.run_baseline import _load_checkpoint, _rebuild_generation_results
from graphinstruct.evaluate import _build_reference_graphs
from graphinstruct.metrics.structural import valid_rate_single


def compute_d3_for_instruction(inst, gen_results):
    """Compute D3 sub-metrics for a single instruction.

    Returns dict with per-sample and aggregate scores, or None if level < 3.
    """
    level = inst.level
    if level < 3:
        return None

    # Get valid generated graphs
    graphs = [r.graph for r in gen_results if r.graph is not None]

    # Extract graph_type from constraints for valid_rate_single
    graph_type = None
    other_constraints = []
    for c in inst.explicit_constraints:
        if c.startswith("graph_type="):
            graph_type = c.split("=", 1)[1]
        else:
            other_constraints.append(c)

    valid_graphs = [
        g
        for g in graphs
        if valid_rate_single(g, graph_type=graph_type, constraints=other_constraints)
    ]

    if not valid_graphs:
        return {
            "instruction_id": inst.id,
            "level": level,
            "n_valid": 0,
            "n_generated": len(graphs),
            "grassmann": 0.0,
            "node_clf_gap": 0.0,
            "emb_mmd": 0.0,
            "aggregate": 0.0,
            "per_sample": [],
        }

    # Build reference graph
    ref_graphs = _build_reference_graphs(inst)
    ref_graph = ref_graphs[0] if ref_graphs else None

    if ref_graph is None:
        return {
            "instruction_id": inst.id,
            "level": level,
            "n_valid": len(valid_graphs),
            "n_generated": len(graphs),
            "grassmann": 0.0,
            "node_clf_gap": 0.0,
            "emb_mmd": 0.0,
            "aggregate": 0.0,
            "per_sample": [],
            "note": "no_reference_graph",
        }

    # Compute D3 sub-metrics in batch (ref GCN trained once, reused across samples)
    gc_scores = [grassmann_coherence(g, ref_graph) for g in valid_graphs]
    ncg_scores = node_clf_gap_batch(valid_graphs, ref_graph)
    emmd_scores = [embedding_mmd(g, ref_graph) for g in valid_graphs]

    per_sample = [
        {
            "grassmann": round(gc, 6),
            "node_clf_gap": round(ncg, 6),
            "emb_mmd": round(emmd, 6),
        }
        for gc, ncg, emmd in zip(gc_scores, ncg_scores, emmd_scores)
    ]

    n = len(valid_graphs)
    avg_gc = sum(gc_scores) / n
    avg_ncg = sum(ncg_scores) / n
    avg_emmd = sum(emmd_scores) / n
    avg_agg = (avg_gc + avg_ncg + avg_emmd) / 3

    return {
        "instruction_id": inst.id,
        "level": level,
        "n_valid": n,
        "n_generated": len(graphs),
        "grassmann": round(avg_gc, 6),
        "node_clf_gap": round(avg_ncg, 6),
        "emb_mmd": round(avg_emmd, 6),
        "aggregate": round(avg_agg, 6),
        "per_sample": per_sample,
    }


def process_checkpoint(ckpt_path: Path, all_instructions):
    """Process one JSONL checkpoint and return D3 results."""
    _, records = _load_checkpoint(ckpt_path)
    if not records:
        print(f"  SKIP: no records in {ckpt_path}")
        return None

    all_results = _rebuild_generation_results(records, all_instructions)

    # Build instruction lookup
    inst_lookup = {}
    for insts in all_instructions.values():
        for inst in insts:
            inst_lookup[inst.id] = inst

    # Only process L3+ instructions
    l3plus_ids = [
        iid
        for iid, results in all_results.items()
        if inst_lookup.get(iid) and inst_lookup[iid].level >= 3
    ]
    l3plus_ids.sort()

    print(f"  {len(l3plus_ids)} instructions at L3+ to evaluate")

    d3_results = []
    t0 = time.time()
    for i, iid in enumerate(l3plus_ids):
        inst = inst_lookup[iid]
        results = all_results[iid]
        result = compute_d3_for_instruction(inst, results)
        if result is not None:
            d3_results.append(result)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(l3plus_ids) - i - 1) / rate
            print(f"    [{i+1}/{len(l3plus_ids)}] {rate:.1f} inst/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0

    # Aggregate per-level summary
    level_summary = {}
    for lvl in [3, 4, 5]:
        items = [r for r in d3_results if r["level"] == lvl]
        if not items:
            continue
        n = len(items)
        level_summary[lvl] = {
            "count": n,
            "grassmann_mean": round(sum(r["grassmann"] for r in items) / n, 6),
            "node_clf_gap_mean": round(sum(r["node_clf_gap"] for r in items) / n, 6),
            "emb_mmd_mean": round(sum(r["emb_mmd"] for r in items) / n, 6),
            "aggregate_mean": round(sum(r["aggregate"] for r in items) / n, 6),
            "grassmann_std": round(_std([r["grassmann"] for r in items]), 6),
            "node_clf_gap_std": round(_std([r["node_clf_gap"] for r in items]), 6),
            "emb_mmd_std": round(_std([r["emb_mmd"] for r in items]), 6),
        }

    return {
        "per_instruction": d3_results,
        "per_level_summary": level_summary,
        "metadata": {
            "checkpoint": str(ckpt_path),
            "backend": "pyg" if _HAS_PYG else ("torch" if _HAS_TORCH else "numpy"),
            "has_numpy": _HAS_NUMPY,
            "has_torch": _HAS_TORCH,
            "has_pyg": _HAS_PYG,
            "elapsed_seconds": round(elapsed, 1),
            "n_instructions": len(d3_results),
        },
    }


def _std(values):
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def main():
    parser = argparse.ArgumentParser(description="Evaluate D3 sub-metrics only")
    parser.add_argument("checkpoints", nargs="+", help="JSONL checkpoint file(s)")
    args = parser.parse_args()

    print(f"D3 backend: numpy={_HAS_NUMPY}, torch={_HAS_TORCH}, pyg={_HAS_PYG}")

    # Load instructions once
    all_instructions = load_all_levels([0, 1, 2, 3, 4, 5])
    total_l3plus = sum(len(v) for k, v in all_instructions.items() if k >= 3)
    print(
        f"Loaded instructions: {total_l3plus} at L3+ (L3:{len(all_instructions[3])}, "
        f"L4:{len(all_instructions[4])}, L5:{len(all_instructions[5])})"
    )

    for ckpt in args.checkpoints:
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            print(f"SKIP: {ckpt_path} not found")
            continue

        print(f"\nProcessing {ckpt_path.name}...")
        result = process_checkpoint(ckpt_path, all_instructions)
        if result is None:
            continue

        out_path = ckpt_path.with_name(ckpt_path.stem + ".d3.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Print summary
        print(f"  Backend: {result['metadata']['backend']}")
        print(f"  Time: {result['metadata']['elapsed_seconds']}s")
        for lvl, s in sorted(result["per_level_summary"].items()):
            print(
                f"  L{lvl} (n={s['count']}): "
                f"grassmann={s['grassmann_mean']:.4f}±{s['grassmann_std']:.4f}  "
                f"ncg={s['node_clf_gap_mean']:.4f}±{s['node_clf_gap_std']:.4f}  "
                f"emb_mmd={s['emb_mmd_mean']:.4f}±{s['emb_mmd_std']:.4f}  "
                f"agg={s['aggregate_mean']:.4f}"
            )
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
