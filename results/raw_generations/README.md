# `raw_generations/` — full LLM outputs (hosted externally)

The complete set of raw LLM generations — the 45-cell capability survey
over 12 LLMs and 4 prompting strategies, plus the discussion-phase runs
(reasoning panel, extended baseline models, held-out-exemplar,
disabled-thinking, and refinement methods; **124 runs in total**) — is
**not bundled** in this repository because it totals **~1.3 GB**. Below are the alternative hosting locations.

## Contents

| Item | Size | Format |
|------|------|--------|
| `<model>-<strategy>.jsonl` × 124 | ~1.3 GB total | One JSON per line; each line = `{instruction_id, level, samples: [{raw_output, graph_serialized, valid, input_tokens, output_tokens, api_calls}, …]}` |
| Sample format | — | 5 samples per instruction × 800 instructions; the 45-cell core survey alone = 180,000 generations |

## Download

### Option 1 — Zenodo (preferred; permanent DOI)

The Zenodo record contains a **complete frozen snapshot** of `results/`
(leaderboards + 124 quality.json + raw_generations/). If you only need
quality scores or leaderboards, the GitHub copy of those files is sufficient;
the Zenodo archive is for raw outputs and long-term archival.

**DOI: [10.5281/zenodo.21596998](https://doi.org/10.5281/zenodo.21596998)** — record page: https://zenodo.org/records/21596998

```bash
wget https://zenodo.org/records/21596998/files/graphinstruct-results.tar.gz
# sha256: ffc238fdfeb33cf7cd4df817c950eab099807333087ae2b4c5903e7ab65a093b
tar -xzf graphinstruct-results.tar.gz -C results/
```

### Option 2 — recompute from scratch

If you have API access to the 12 LLMs surveyed, you can recompute the raw
generations yourself using `scripts/run_baseline.py`. See
[`docs/REPRODUCE.md`](../../docs/REPRODUCE.md) Tier C for the recipe and cost
estimate (~$600, ~7 days wall-clock).

## When you need this

| Task | Need raw generations? |
|------|-----------------------|
| Verify a single paper number | No — `results/quality/` is sufficient |
| Re-run the D5 robustness ablation | No — uses `quality/` only |
| Inspect what a specific (model, instruction) output was | Yes |
| Re-evaluate with a new metric | Yes |
| Train a downstream model on these generations | Yes |

## License

Raw generations are released under **CC-BY-4.0**. Note that some upstream
LLMs (Claude, GPT, Qwen) impose their own usage terms on outputs — see each
provider's TOS if you intend commercial redistribution.
