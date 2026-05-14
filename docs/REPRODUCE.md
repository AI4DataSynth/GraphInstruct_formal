# Reproduction Guide


> Maps each main-paper number to the exact command that produces it. You should be able to recover every claim in §5 (Results) and Appendix C, F, G of the paper using the recipes below.

### Three reproduction tiers

We provide three tiers of reproduction depending on whether you want to (A) verify the data + metrics, (B) re-evaluate cached LLM outputs, or (C) re-generate from scratch.

| Tier | What you reproduce | Compute | API cost | Time |
|------|-------------------|---------|----------|------|
| **A** | Data integrity, parser, metric correctness, D5 robustness | CPU | $0 | ~5 min |
| **B** | Quality / Combined / S_final scores from cached generations | CPU + GPU (D3) | $0 | ~45 min |
| **C** | Full 45-cell survey over 12 LLMs and 4 prompting strategies | API | ~$600 | ~7 days |

### Tier A: Verify data and metrics (CPU only, no API)

```bash
# 1. Run all 549 unit tests
python -m unittest discover -v -s tests
#    → Should report: OK (549 tests)

# 2. Round-trip parser test on the 1,582 reference solutions
python -m unittest tests.test_parser -v
#    → Verifies every reference passes parse → serialize → re-parse

# 3. D5 exponential-scale robustness ablation (paper Appendix C, Table 3)
python scripts/d5_robustness.py
#    → Should print: ρ ∈ [0.966, 1.000] across the 3×3 (s_T, s_A) grid
```

**Expected output for step 3 (D5 robustness)**:

```
   s_T    s_A    Spearman     Kendall    Top-1 stable   Top-5 jaccard
   500      1      0.9663      0.8687              no           0.429
   500      2      0.9663      0.8687              no           0.429
   500      4      0.9663      0.8687              no           0.429
  1000      1      1.0000      1.0000             yes           1.000
  1000      2      1.0000      1.0000             yes           1.000
  1000      4      1.0000      1.0000             yes           1.000
  2000      1      0.9888      0.9333              no           1.000
  2000      2      0.9888      0.9333              no           1.000
  2000      4      0.9888      0.9333              no           1.000
```

This step uses the bundled curated quality files under `results/quality/*.quality.json`; the script also supports the legacy flat layout `results/*.quality.json`.

### Tier B: Re-evaluate cached generations

This repository already bundles curated `results/quality/*.quality.json` files and the leaderboard CSVs used by the paper tables. Re-evaluating from raw generations requires downloading the cached `results/*.jsonl` raw generation outputs from the repository (~1 GB). Then:

```bash
# 1. Re-evaluate a single (model, strategy) pair end-to-end
python scripts/eval_checkpoint.py \
    --jsonl results/gpt4omini-zero-shot.jsonl \
    --output results/gpt4omini-zero-shot.quality.json
#    → Computes D1 + D4 + D5 (CPU); D2 + D3 require [full] install

# 2. Generate the per-level Quality leaderboard
python scripts/visualize_results.py --leaderboard
#    → Reproduces Tab. 1 (top-15) and full per-level tables in Appendix G

# 3. Compute Pareto frontier and S_final (paper §5.3)
python scripts/visualize_results.py --pareto
#    → Reproduces Fig. 8 and Tab. 2 (efficiency leaderboard)
```

### Tier C: Full re-generation from scratch (~$600, ~7 days wall-clock)

> **Important — `max_tokens` settings.** The paper uses `max_tokens=4096` for `gpt-3.5-turbo` (its API hard limit) and `max_tokens=16384` for all 11 other models. `run_baseline.py` auto-selects this per-model value when `--max-tokens` is not passed, so the commands below need no extra flag. If you override with a different value, the runner prints a `[WARN]` and your numbers may not match the leaderboard (long L4/L5 outputs get truncated under 4096).

```bash
# 1. Run baseline survey for one model (5 generations × 800 instructions)
export OPENAI_API_KEY=sk-...
python scripts/run_baseline.py --model gpt-4o-mini --strategy zero-shot --resume
python scripts/run_baseline.py --model gpt-4o-mini --strategy few-shot  --resume
python scripts/run_baseline.py --model gpt-4o-mini --strategy zero-cot  --resume
python scripts/run_baseline.py --model gpt-4o-mini --strategy few-cot   --resume

# 2. Repeat for each of the 12 models (loop over the model list in the paper)

# 3. Run improvement methods on the 3 target models
python scripts/run_baseline.py --model gpt-4o-mini --strategy combined --resume
python scripts/run_baseline.py --model deepseek-v3 --strategy combined --resume
python scripts/run_baseline.py --model qwen3.5-35b --strategy combined --resume

# 4. Aggregate and visualize (same commands as Tier B above)
```

**Note on `--resume`**: always pass `--resume` to avoid wiping prior partial output. Without it, `run_baseline.py` re-creates the output file from scratch.

### Mapping main-paper numbers to commands

| Paper claim | Source data | Command | File regenerated |
|-------------|-------------|---------|------------------|
| Tab. 1 (top-15 leaderboard) | `results/quality/*.quality.json` × 45 | `python scripts/visualize_results.py --leaderboard` | `results/leaderboards/tab1_quality_top15.csv` |
| Fig. 5 (E5 rounds curve) | E5 ablation outputs | `python scripts/run_baseline.py --ablation e5_rounds` then visualize | `figures/fig5_e5_rounds.png` |
| Fig. 6 (E6 feedback granularity) | E6 ablation outputs | similar to E5 | `figures/fig6_e6_feedback.png` |
| Fig. 8 (Pareto frontier) | `results/quality/*.quality.json` × 45 | `python scripts/visualize_results.py --pareto` | `figures/fig8_pareto.png` |
| Tab. 3 (D5 robustness) | `results/quality/*.quality.json` | `python scripts/d5_robustness.py` | stdout |
| F.4 (L4 retrieval grounding evidence) | GPT-4o-mini zero-shot vs few-shot quality.json | `diff` of `gpt4omini-{zero,few}-shot.quality.json` per-level | — |
| Case study L2-143 (Fig. 12 in App. F) | per-instance D1 outputs | `python scripts/case_study_l2.py --inst-id L2-143` | `figures/fig_case_study_L2.png` |

### Common reproduction issues

- **D3 metric throws CUDA OOM**: lower `--batch-size` from default 32 to 16 or 8.
- **Different result on `d5_robustness.py`**: confirm the scoring weights in `graphinstruct/scoring.py` match those reported in paper §3 (D5 weights at L0/L1/L2/L3/L4/L5 are 0.30/0.15/0.15/0.20/0.15/0.20).
- **API rate limiting**: `run_baseline.py` retries with exponential backoff; large surveys may take longer than the wall-clock estimates above. Pass `--max-concurrent 4` to limit parallelism.
