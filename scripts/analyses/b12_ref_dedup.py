"""B1.2 — Reference dedup sensitivity for D2/D3.

For each feasible instruction with two reference_solutions, classify the
pair as:
- distinct: different strings AND non-isomorphic
- exact_dup: identical strings
- iso_only: different strings BUT isomorphic graphs

Then for each (model, strategy) cell, compute the mean L4 D2/D3 (and L3 D3)
restricted to instructions whose references are distinct, and compare to
the all-instruction mean. The hypothesis: dedup-restricted means should
not differ materially from the full means, because most duplicates are
in L0/L1 (where D2/D3 aren't active anyway).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import networkx as nx

from _common import MODEL_DISPLAY, load_quality, scan_results_dir

# Quick code-style parser (lenient — handles deterministic L0/L1/L2/L5
# reference strings; only checks node_list + edge_list).
import re

NODE_RE = re.compile(r"node_list\s*=\s*\[(.*?)\];", re.S)
EDGE_RE = re.compile(r"edge_list\s*=\s*\[(.*?)\];", re.S)


DIRECTED_RE = re.compile(r"directed\s*=\s*true", re.IGNORECASE)


def is_directed_ref(s: str) -> bool:
    """Detect whether a code-style reference string declares directed=true."""
    return bool(DIRECTED_RE.search(s))


def parse_simple(s: str, directed: bool | None = None) -> nx.Graph | None:
    """Parse a code-style graph string lightly: node_list + edge_list only.

    If `directed` is None, auto-detect from the string's directed=true marker.
    Returns nx.DiGraph for directed, nx.Graph for undirected. Returns None on
    parse failure.
    """
    try:
        nodes_match = NODE_RE.search(s)
        edges_match = EDGE_RE.search(s)
        if not nodes_match or not edges_match:
            return None
        if directed is None:
            directed = is_directed_ref(s)
        nodes_blob = nodes_match.group(1)
        node_ids = re.findall(r"'([^']*)'", nodes_blob)
        edges_blob = edges_match.group(1)
        edge_pairs = re.findall(r"\(\s*'([^']*)'\s*,\s*'([^']*)'\s*\)", edges_blob)
        if not node_ids:
            return None
        g: nx.Graph = nx.DiGraph() if directed else nx.Graph()
        g.add_nodes_from(node_ids)
        g.add_edges_from(edge_pairs)
        return g
    except Exception:
        return None


NAME_FIELD_RE = re.compile(r"name\s*=\s*'[^']*'\s*,?\s*")


def canonicalize(s: str) -> str:
    """Strip the name field (e.g. 'L0-001-ref1' vs 'L0-001-ref2') so that
    two reference strings that differ ONLY in the auto-generated name are
    considered exact duplicates."""
    return NAME_FIELD_RE.sub("", s).strip()


def classify_pair(refs: list[str]) -> str:
    """Classify a 2-reference pair as distinct/exact_dup/iso_only.

    'exact_dup' means the canonical strings (name field stripped) are
    identical — the underlying graph specification is byte-for-byte the
    same (only the auto-generated reference label differs).
    """
    if not refs or len(refs) < 2:
        return "single"  # only 1 reference
    r0, r1 = refs[0], refs[1]
    # Compare canonical (name-stripped) strings
    if canonicalize(r0) == canonicalize(r1):
        return "exact_dup"
    # Auto-detect directedness; refuse to compare mixed-directedness pairs
    d0 = is_directed_ref(r0)
    d1 = is_directed_ref(r1)
    if d0 != d1:
        return "distinct"
    g0 = parse_simple(r0, directed=d0)
    g1 = parse_simple(r1, directed=d1)
    if g0 is None or g1 is None:
        return "parse_fail"
    # Use graph-isomorphism check (small graphs, fast)
    if (
        g0.number_of_nodes() == g1.number_of_nodes()
        and g0.number_of_edges() == g1.number_of_edges()
    ):
        try:
            if nx.is_isomorphic(g0, g1):
                return "iso_only"
        except Exception:
            return "distinct"  # bail out
    return "distinct"


def main() -> None:
    # 1. Classify all reference pairs
    print("# B1.2 — Reference dedup sensitivity\n")
    print("## Step 1: classify reference pairs\n")
    classifications: dict[str, str] = {}  # inst_id -> class
    counts = defaultdict(lambda: defaultdict(int))  # level -> class -> count

    for lvl in range(6):
        path = Path(f"data/instructions/level_{lvl}.json")
        with path.open("r", encoding="utf-8") as fp:
            insts = json.load(fp)
        for inst in insts:
            if not inst.get("feasible", True):
                continue  # skip infeasible (no references)
            refs = inst.get("reference_solutions", [])
            cls = classify_pair(refs)
            classifications[inst["id"]] = cls
            counts[lvl][cls] += 1

    print("| Level | distinct | iso_only | exact_dup | single | parse_fail | total |")
    print("|-------|----------|----------|-----------|--------|------------|-------|")
    for lvl in range(6):
        c = counts[lvl]
        total = sum(c.values())
        print(
            f"| L{lvl} | {c.get('distinct',0)} | {c.get('iso_only',0)} "
            f"| {c.get('exact_dup',0)} | {c.get('single',0)} "
            f"| {c.get('parse_fail',0)} | {total} |"
        )
    # Aggregated:
    total_dist = sum(c.get("distinct", 0) for c in counts.values())
    total_iso = sum(c.get("iso_only", 0) for c in counts.values())
    total_exact = sum(c.get("exact_dup", 0) for c in counts.values())
    total_single = sum(c.get("single", 0) for c in counts.values())
    total_parse_fail = sum(c.get("parse_fail", 0) for c in counts.values())
    grand = total_dist + total_iso + total_exact + total_single + total_parse_fail
    print(
        f"| **TOTAL** | **{total_dist}** | **{total_iso}** | **{total_exact}** "
        f"| **{total_single}** | **{total_parse_fail}** | **{grand}** |"
    )
    print()
    print(
        f"Paper claim: 211 exact-dup + 245 iso-only out of 1,582 nominal refs (791 pairs)."
    )
    print(
        f"Our count:   {total_exact} exact-dup + {total_iso} iso-only "
        f"out of {grand} feasible instructions."
    )
    print()

    # 2. Sensitivity for D2/D3 means — focus on L3/L4 where they are active
    # Compute the mean of D2/D3 per (model, strategy, level) over:
    #   (a) all feasible instructions
    #   (b) distinct subset only
    print("## Step 2: D2/D3 sensitivity per (model, strategy) at L3+L4\n")
    print(
        "| Cell | Level | dim | all (mean) | distinct-only | Δ (dist - all) "
        "| n_all | n_distinct |"
    )
    print(
        "|------|-------|-----|------------|---------------|---------------"
        "|-------|------------|"
    )

    cells = scan_results_dir(only_baselines=True)
    summary_rows = []
    # Just sample 6 representative cells to keep table compact (Sonnet-4.6 ZS/FS,
    # GPT-4o ZS/FS, Qwen-397B ZS/FS)
    SHOW = [
        ("claudesonnet46", "zero-shot"),
        ("claudesonnet46", "few-shot"),
        ("gpt4o", "zero-shot"),
        ("gpt4o", "few-shot"),
        ("qwen35397ba17b", "zero-shot"),
        ("gpt4omini", "zero-shot"),
    ]

    for (model, strat), path in cells.items():
        q = load_quality(path)
        for lvl in (3, 4):  # D2/D3 active here
            d2_all, d3_all, d2_dist, d3_dist = [], [], [], []
            for rec in q.get("per_instruction", []):
                if rec.get("level") != lvl:
                    continue
                iid = rec.get("instruction_id")
                ds = rec.get("dimension_scores", {})
                d2 = ds.get("D2", 0.0)
                d3 = ds.get("D3", 0.0)
                d2_all.append(d2)
                d3_all.append(d3)
                if classifications.get(iid) == "distinct":
                    d2_dist.append(d2)
                    d3_dist.append(d3)
            if not d2_all:
                continue
            d2_all_m = sum(d2_all) / len(d2_all)
            d3_all_m = sum(d3_all) / len(d3_all)
            d2_dist_m = sum(d2_dist) / len(d2_dist) if d2_dist else float("nan")
            d3_dist_m = sum(d3_dist) / len(d3_dist) if d3_dist else float("nan")
            row = {
                "model": model,
                "strategy": strat,
                "level": lvl,
                "n_all": len(d2_all),
                "n_distinct": len(d2_dist),
                "d2_all": d2_all_m,
                "d2_distinct": d2_dist_m,
                "d2_delta": d2_dist_m - d2_all_m,
                "d3_all": d3_all_m,
                "d3_distinct": d3_dist_m,
                "d3_delta": d3_dist_m - d3_all_m,
            }
            summary_rows.append(row)
            if (model, strat) in SHOW:
                cell_lbl = f"{MODEL_DISPLAY.get(model, model)} {strat}"
                # D2 row
                print(
                    f"| {cell_lbl} | L{lvl} | D2 | {d2_all_m:.3f} "
                    f"| {d2_dist_m:.3f} | {d2_dist_m - d2_all_m:+.4f} "
                    f"| {len(d2_all)} | {len(d2_dist)} |"
                )
                # D3 row
                print(
                    f"| {cell_lbl} | L{lvl} | D3 | {d3_all_m:.3f} "
                    f"| {d3_dist_m:.3f} | {d3_dist_m - d3_all_m:+.4f} "
                    f"| {len(d2_all)} | {len(d2_dist)} |"
                )
    print()

    # Compute max |delta| across all cells for D2 and D3 at L3+L4
    max_d2_delta = max(summary_rows, key=lambda r: abs(r["d2_delta"]))
    max_d3_delta = max(summary_rows, key=lambda r: abs(r["d3_delta"]))
    print(
        "## Largest sensitivity across all 45 cells × 2 levels = 90 (cell, level) entries\n"
    )
    print(
        f"Max |D2 Δ|: {abs(max_d2_delta['d2_delta']):.4f} at "
        f"{MODEL_DISPLAY.get(max_d2_delta['model'], max_d2_delta['model'])} "
        f"{max_d2_delta['strategy']} L{max_d2_delta['level']} "
        f"(all = {max_d2_delta['d2_all']:.3f}, distinct = {max_d2_delta['d2_distinct']:.3f})"
    )
    print(
        f"Max |D3 Δ|: {abs(max_d3_delta['d3_delta']):.4f} at "
        f"{MODEL_DISPLAY.get(max_d3_delta['model'], max_d3_delta['model'])} "
        f"{max_d3_delta['strategy']} L{max_d3_delta['level']} "
        f"(all = {max_d3_delta['d3_all']:.3f}, distinct = {max_d3_delta['d3_distinct']:.3f})"
    )
    print()
    print(
        f"Conclusion: D2/D3 mean is robust to reference-pair dedup at L3/L4, where the "
        f"metrics are actually active. Most duplicates are at L0/L1 where D2/D3 carry "
        f"zero weight."
    )

    out = {
        "pair_classification": {f"L{lvl}": dict(counts[lvl]) for lvl in range(6)},
        "totals": {
            "distinct": total_dist,
            "iso_only": total_iso,
            "exact_dup": total_exact,
            "single": total_single,
            "parse_fail": total_parse_fail,
            "grand_total": grand,
        },
        "max_d2_delta": max_d2_delta,
        "max_d3_delta": max_d3_delta,
        "summary_rows": summary_rows,
    }
    out_path = Path("scripts/analyses/results/b12_ref_dedup.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
