# `raw_generations/` — full LLM outputs (hosted externally)

The complete set of raw LLM generations from the 12-LLM × 4-strategy capability
survey is **not bundled** in this repository because it totals **~1 GB**. Below
are the alternative hosting locations.

## Contents

| Item | Size | Format |
|------|------|--------|
| `<model>-<strategy>.jsonl` × 45 | ~900 MB total | One JSON per line; each line = `{instruction_id, level, samples: [{raw_output, graph_serialized, valid, input_tokens, output_tokens, api_calls}, …]}` |
| Sample format | — | 5 samples per instruction × 800 instructions × 45 cells = 180,000 generations |

## Download

### Option 1 — Zenodo (preferred; permanent DOI)

The Zenodo record contains a **complete frozen snapshot** of `results/`
(leaderboards + 45 quality.json + raw_generations/). Reviewers who only need
quality scores or leaderboards can use the GitHub copy of those files; the
Zenodo archive is for raw outputs and long-term archival.

*This URL will be filled in upon Zenodo upload (see
`PLACEHOLDER_ZENODO_DOI` in repo metadata).*

```bash
# Replace PLACEHOLDER_RECORD with the actual Zenodo record ID once uploaded
wget https://zenodo.org/records/PLACEHOLDER_RECORD/files/graphinstruct-results.tar.gz
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
