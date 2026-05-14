# `results/` — pre-computed experimental data

This directory contains the computed evaluation outputs that back the numbers
in the GraphInstruct paper. You can verify any paper number by inspecting
the JSON / CSV files here directly, without re-running the (expensive) LLM
generation pipeline.

```
results/
├── leaderboards/                 ← 3 CSV files, one per main-paper table
├── quality/                      ← 45 per-cell quality scores (12 LLMs × 4 strategies + Sonnet-4 ZS)
└── raw_generations/              ← (placeholder) raw model outputs, hosted on Zenodo
```

---

## `leaderboards/` — paper-table CSVs

| File | Maps to paper | Contents |
|------|---------------|----------|
| `tab1_quality_top15.csv` | Table 1 (top-15 capability leaderboard) | Rank, model, strategy, total Quality, per-level scores L0–L5, TPV |
| `tab2_pareto_efficiency.csv` | Table 2 (efficiency-aware leaderboard) | S_final rank, Quality, TPV, Q per kTPV |
| `tab3_d5_robustness.csv` | Appendix C, Table 3 (D5 exponential-scale robustness) | (s_T, s_A) grid × Spearman ρ, Kendall τ, top-1 stability, top-5 Jaccard |

These are deterministic re-derivations from the `quality/` JSONs — you can
re-run `python scripts/_build_leaderboards.py` to regenerate.

---

## `quality/` — per-cell evaluation outputs (45 files)

Each file is named `<model>-<strategy>.quality.json` and contains:

```jsonc
{
  "per_instruction": [           // 800 entries (one per instruction)
    {
      "instruction_id": "L0-001",
      "level": 0,
      "d1_valid_rate": 1.0,
      "d4_instruction_score": 1.0,
      "d5_tokens_per_valid": 195.6,
      "dimension_scores": {
        "D1": 0.82, "D2": 0.0, "D3": 0.0, "D4": 1.0, "D5": 0.876
      },
      "level_score": 0.974
    },
    ...
  ],
  "per_level_scores": {"0": 0.78, "1": 0.86, ...},
  "total_score": 0.7287,
  "final_score": 0.8377,
  "metadata": {"model": "gpt-4o-mini", "strategy": "zero-shot", ...}
}
```

The 45 cells span:

- **11 fully-evaluated models** × **4 prompting strategies** = **44 cells**
- **+ Sonnet-4** evaluated on **zero-shot only** (efficiency-baseline reference) = **1 cell**

Total: **45 cells**.

You can verify any single number in paper §5 by opening the corresponding
file. For example, paper §5.4 reports GPT-4o-mini L4 zero-shot Q = 0.744 — open
`quality/gpt4omini-zero-shot.quality.json`, look at `per_level_scores["4"]`.

---

## `raw_generations/` — full LLM outputs (Zenodo)

The complete 180,000-output set of raw LLM generations (45 cells × 800
instructions × 5 samples) totals **~1 GB** and is **not bundled** in this
repository because it would push the repo past GitHub's recommended size.

The raw generations live on **Zenodo** as part of a complete frozen snapshot
of this `results/` directory:

- **Zenodo DOI**: *(to be filled in upon upload — see
  `raw_generations/README.md` for the wget recipe)*

The Zenodo archive is a **superset** of what GitHub serves: it includes the
leaderboard CSVs and 45 quality.json files (mirrored from this repo) plus the
raw_generations/*.jsonl files (GitHub-only-omitted). This dual hosting gives
users a fast on-GitHub path for verifying single numbers and a single
DOI-citable archive for long-term reproducibility.

If you only want to verify paper numbers, the `quality/` JSON files above are
sufficient — they contain the per-instance scores aggregated from the raw
generations.
