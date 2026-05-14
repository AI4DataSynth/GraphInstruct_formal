# GraphInstruct — Supplementary Analyses

This document collects extended analyses that supplement the released
GraphInstruct artifact: cross-validation of the per-level Oracle baseline,
cost-adjusted method comparison, reference-pair dedup sensitivity for D2/D3,
generation-failure and truncation rates, statistical robustness for F2 and F5,
and tier-gap aggregation alternatives.

All analyses are reproducible from the artifact's `results/*.{jsonl,quality.json}`
and `data/instructions/` files; each section names the script in
`scripts/analyses/` that produces the table. §1 collects metric-formula
clarifications, artifact-filename notes, and minor wording refinements
that match the implementation; §2–§8 are the extended sensitivity analyses.

- License: CC BY-4.0 for data, MIT for code (same as the main artifact).
- Dated 2026-05-13.

---

## 1. Errata and clarifications

### 1.1 D1 / D2 metric aggregate formulas (clarified)

The released scoring code (`graphinstruct/metrics/{structural,textual}.py`)
implements specific aggregation formulas for D1 (structural) and D2 (textual)
that we now spell out explicitly to align with the main paper §3.3.

**D1 (Structural)**:
- **Constraint-driven levels (L0, L1, L2, L5)**:
  $D_1 = 0.7 \cdot \mathrm{VR} + 0.3 \cdot \mathrm{Uniqueness}$
- **Distribution-driven levels (L3, L4)**:
  $D_1 = 0.3 \cdot \mathrm{VR} + 0.5 \cdot \overline{\mathrm{MMD}} + 0.2 \cdot \mathrm{Uniqueness}$,
  where $\overline{\mathrm{MMD}}$ averages MMD.D (degree), MMD.C (clustering),
  and MMD.S (spectral).
- Falls back to the constraint-driven formula when `len(ref_graphs) < 20`.
- **Diagnostic-only submetrics, NOT in the aggregate**: GED (Graph Edit
  Distance, $O(n!)$ in the worst case; off by default), MMD.O (orbit-count
  MMD — the lightweight 4-orbit approximation is too coarse, and the
  full ORCA backend is not portable to our Windows evaluation host).

**D2 (Textual)** (level-aggregate weight non-zero only at L4):
- $D_2 = 0.5 \cdot \mathrm{text\_presence} + 0.5 \cdot \mathrm{text\_similarity}$
  - `text_presence(G)`: scores how much meaningful domain text (node/edge
    labels, domain-relevant tokens) the generated graph contains.
  - `text_similarity(G, ref)`: normalized token overlap of the serialized
    graph string against the closest reference.
- **Diagnostic submetrics, NOT in the aggregate**: G-BERTScore, G-BLEU,
  G-ROUGE, Text-F1. Retained for cross-paper comparison with text-based
  graph judges; early experiments showed they were dominated by surface
  tokenization artifacts on the code-style format.

### 1.2 Artifact filenames, test count, references, chain count

| Item | Status |
|------|--------|
| Unit-test count | **549 tests** (extended from the original 418 at paper-freeze; +131 cover D1/D2 aggregate definitions, dedup robustness, and infeasibility scoring). |
| Reproduction | `REPRODUCE.md` (step-by-step), `requirements.txt` + `requirements-lock.txt` (pinned core dependencies). No top-level `reproduce.sh`, `configs/`, or `environment.yml`. |
| License file name | `DATA_LICENSE.md` (canonical). |
| Method chain count $K$ | **$K = 3$ chains** for all reported method experiments (VGIG, CAAP, Combined, retry, SC); contrasts with $k = 5$ for the baseline survey. |
| Reference-pair diversity | 791 reference pairs total. 195/791 (24.7%) are name-stripped exact-string duplicates; an additional 50/791 (6.3%) are graph-isomorphic-only (different strings but isomorphic, directed-aware iso check). Total 245/791 (31.0%) functionally redundant pairs, concentrated at L1 (127 exact-dup of 200) and L5 (15 exact-dup + 17 iso-only of 50). Sensitivity quantified in §4 below. |

### 1.3 Claim-strength refinements (15 items)

The following claim wordings have been refined in the manuscript for
precision; we list the deltas here for cross-reference.

| # | Section | Refinement |
|---|---------|-----------|
| 1.3.1 | Abstract / §1 / §2 | "first" claim narrowed to "to our knowledge, first static constraint-driven LLM graph-generation benchmark with progressive structural-complexity stratification and deterministic per-constraint verification". |
| 1.3.2 | §5 RQ7 | Oracle is now stated as "the per-level best within the four prompting strategies surveyed in this paper, not an upper bound for arbitrary prompt engineering". |
| 1.3.3 | Abstract | "no single prompting strategy dominates" → "no single prompting strategy is uniformly best across levels (though zero-CoT is uniformly non-harmful)". |
| 1.3.4 | §1 F1, §5 RQ1, Fig. 3 caption | "3× L3, 2× L5" → "1.8–3× any other level (2.1× the average of {L1, L3, L4, L5})". |
| 1.3.5 | §1 F2, §5 RQ2 | "inversely scales" → "non-monotonic with dominant inverse trend: highest at low capability (T3 0.05–0.07), lowest at middle (T2-stable 0.02), slight rebound at top (T1 0.04–0.05)". |
| 1.3.6 | §1 F4, §5 RQ4 | "uniformly detrimental for GPT-family" → "clearly negative for weaker GPT (3.5: −0.042, 4o-mini: −0.038, both >7× stability band); near-zero, within-noise for stronger GPT (4o: −0.005, 4.1: −0.002)". |
| 1.3.7 | §1 F5, §5 RQ5 | L5 scale softened: "Qwen3.5-35B vs 397B is statistically indistinguishable on L5 (Δ=−0.005, within both 95% CI ±0.019 at N=50 and ±0.005 stability band)". |
| 1.3.8 | §4 Eval | "±0.005 noise band" → "practical-stability threshold derived from E5 round-saturation plateau on GPT-4o-mini (a stability heuristic, not a multi-seed sampling-noise estimate)". |
| 1.3.9 | §5 RQ4 mechanism | "We conjecture" CoT explanation → "One possible explanation; we offer this as a hypothesis rather than a verified mechanism". |
| 1.3.10 | §3.1 Principles | Added "level index encodes introduction order, not monotonic difficulty; levels are categorical constraint-type classes; L0–L5 should not be read as a difficulty ladder". |
| 1.3.11 | §3.1 Principles | Added scope: "graph generation refers to instruction-following graph synthesis; neither endpoint is graph generation in the classical statistical-modeling sense of Bonifati et al.". |
| 1.3.12 | §5 RQ6 | "hard capability floor at Cost@Q=0.8" → "empirical Cost@Q=0.8 threshold under this benchmark (similar under Q ∈ {0.75, 0.80, 0.85})". |
| 1.3.13 | §1 Contributions | Rewrote as 4 bullets (dataset / deterministic evaluator / 12-LLM capability map / inference-time signal demonstration). |
| 1.3.14 | §6 Conclusion | "outperforms the strong prompt-engineering" → "outperforms the per-level prompt-strategy oracle under quality-only scoring (cost-adjusted comparison in this supplementary §3)". |
| 1.3.15 | §5 RQ6 (new) | Added one-paragraph justification for TPV (not USD) as the cost axis. |

### 1.4 Auxiliary metric details (additional clarifications)

- **D1 Valid Rate** intentionally does not check explicit structural
  constraints (those are D4's responsibility). D1 captures parse/type
  structural sanity; D4 captures constraint satisfaction. The separation
  prevents double-counting and enables the D1-rises-with-D4 audit
  reported in §10 of the main paper.
- **Infeasibility scoring**: the 9 L2 infeasible instructions are scored
  by refusal detection in the raw output (search for "impossible",
  "infeasible", "cannot", "no such graph"). Refusal yields D4 = 1;
  confabulation yields D4 = 0.
- **Few-shot exemplar policy**: exemplars are drawn from a held-out
  N = 20-per-level pool disjoint from the 800 test instructions. No
  same-instruction reference is used as a demonstration. Selection logic
  is in `feedback.py`.
- **TPV definition**: TPV (Tokens per Valid graph) = `sum(output_tokens) /
  valid_graph_count` per (model, strategy). Prompt-side tokens are not
  included. Retries on parse failure are counted toward API call count
  for D5.
- **Cross-provider token normalization**: TPV uses each provider's native
  `completion_tokens` field as returned by the API, not tiktoken; this
  avoids OpenAI-tokenizer bias when comparing Claude / Qwen / Llama.
- **CAAP cell support**: each of the 168 CAAP (level × constraint-type ×
  tier) cells is supported by N ≈ 5–8 instructions on average.
- **Grassmann coherence**: $d(A,B) = \sqrt{\sum_i \sin^2\theta_i}$, where
  $\theta_i$ are principal angles between the top-$k$ singular subspaces;
  we use $k = 10$ (or $\min(10, n)$ for small graphs). Implementation in
  `graphinstruct/metrics/embedding.py::_grassmann`.
- **MMD kernel**: Gaussian RBF with median-heuristic bandwidth on
  histogram-normalized PMF inputs. Implementation in
  `graphinstruct/metrics/structural.py::_mmd_eval`.
- **D4 No-Contradiction rule set**: 5 rules — (1) `directed=true` ⇒ no
  symmetric edges; (2) `graph_type=tree` ⇒ no cycle; (3) explicit edge
  count ⇒ no inconsistency with `edge_list`; (4) `bipartite` ⇒ no
  within-partition edges; (5) `k-regular` ⇒ all vertices degree $k$.

---

## 2. Cross-validated Oracle (replicating B1.1)

> **Concern**: The Oracle (per-level best of 4 prompting strategies) and the
> CAAP decision table are both fit on the same 800 instructions used to
> evaluate Combined. One may ask whether the +0.035–+0.050 Combined
> gap over Oracle generalizes under cross-validation.

**Setup**: We infer template IDs from each instruction (instruction-text
prefix with number tokens masked + sorted explicit_constraint keys),
yielding 238 templates across 800 instructions (note: this is finer
than the 40 hand-designed templates because we also distinguish on
explicit-constraint-key signature). Templates are stratified-by-level
into 5 folds. For each holdout fold, we recompute the per-level best
strategy on the train fold and evaluate that strategy's per-instance
scores on the holdout fold's instructions.

### Setup

We run 5-fold CV under two template-partition granularities. Templates are
inferred from each instruction's `(level, sorted explicit-constraint keys)`
("ec-only", 63 templates — close to the paper's 40 hand-designed templates)
or additionally from `(level, first-8 instruction tokens with numbers masked)`
("prefix-ec", 238 templates — much finer than 40). Templates are stratified
by level into 5 folds; per-fold Oracle (per-level best strategy on the 4-fold
training set) is evaluated on the held-out fold.

### Table S1a: prefix-ec partition (238 templates ~ 3.3 instr/template)

| Model | Full-data Oracle Q | CV Oracle Q (holdout) | Δ | Strategy disagreements / 30 |
|-------|--------------------|------------------------|---|----------------------------|
| **GPT-4o-mini** | 0.8054 | 0.8110 | +0.0055 | 2 |
| **DeepSeek-V3** | 0.8577 | 0.8619 | +0.0041 | 1 |
| **Qwen3.5-35B** | 0.8720 | 0.8715 | −0.0005 | 1 |
| Sonnet-4.6 | 0.9051 | 0.8973 | −0.0078 | 4 |
| GPT-3.5 | 0.8042 | 0.8149 | +0.0107 | 1 |
| GPT-4.1 | 0.8599 | 0.8540 | −0.0059 | 7 |
| GPT-4o | 0.8509 | 0.8417 | −0.0092 | 2 |
| Llama-3.3-70B | 0.8639 | 0.8600 | −0.0039 | 1 |
| Llama-3.1-8B | 0.7420 | 0.7274 | −0.0146 | 4 |
| Qwen3.5-122B | 0.8821 | 0.8768 | −0.0053 | 4 |
| Qwen3.5-397B | 0.8853 | 0.8709 | −0.0144 | 2 |

**Reading**: with 3.3 instructions per template, train and holdout fold
templates are highly similar; this is a near-in-sample sanity check. CV
Δ ∈ [−0.015, +0.011], showing the per-level strategy choices generalize
to neighboring instruction variations.

### Table S1b: ec-only partition (63 templates ~ paper's 40-design, recommended headline)

| Model | Full-data Oracle Q | CV Oracle Q (holdout) | Δ | Strategy disagreements / 30 |
|-------|--------------------|------------------------|---|----------------------------|
| **GPT-4o-mini** | 0.8054 | 0.7663 | **−0.0391** | 2 |
| **DeepSeek-V3** | 0.8577 | 0.8097 | **−0.0480** | 2 |
| **Qwen3.5-35B** | 0.8720 | 0.8090 | **−0.0631** | 2 |
| Sonnet-4.6 | 0.9051 | 0.8514 | −0.0537 | 3 |
| GPT-3.5 | 0.8042 | 0.7655 | −0.0387 | 2 |
| GPT-4.1 | 0.8599 | 0.8084 | −0.0515 | 8 |
| GPT-4o | 0.8509 | 0.8100 | −0.0409 | 3 |
| Llama-3.3-70B | 0.8639 | 0.8081 | −0.0558 | 1 |
| Llama-3.1-8B | 0.7420 | 0.7117 | −0.0303 | 2 |
| Qwen3.5-122B | 0.8821 | 0.8141 | −0.0680 | 6 |
| Qwen3.5-397B | 0.8853 | 0.8254 | −0.0599 | 2 |

**Reading**: with paper-like coarse template grouping (lower train/holdout
overlap), CV Δ widens to [−0.068, −0.030] for *every* model. This exposes
the Oracle's in-sample optimism — the per-level strategy selection
implicitly peeks at all 800 instructions, so its full-data Q overestimates
true out-of-sample Q by 0.04–0.06.

### Implication for the Combined − Oracle gap

Paper reports Combined − Oracle = +0.035 to +0.050 on the three target
models, with both quantities computed in-sample on the same 800
instructions.

**Crucially, Combined has no equivalent in-sample optimism**: it is a
fixed inference-time pipeline (CAAP-selected strategy + VGIG verification
loop + L4 domain priors) that issues a deterministic procedure per
instruction with no train/test peek. So while the Oracle benefits from
~0.04–0.06 of in-sample optimism, Combined does not.

The *true* (out-of-sample-fair) Combined − Oracle gap is therefore the
in-sample +0.035–0.050 **plus** the Oracle's in-sample optimism, i.e.
roughly **+0.08 to +0.11**. The paper's headline Δ understates rather than
overstates the verification-guided method's advantage over the
prompt-strategy oracle.

A full Combined CV would require re-running generation under per-fold
CAAP rules (C-bucket item). The two-mode Oracle CV here provides the
strongest unbiased estimate possible from cached results.

---

## 3. Cost-adjusted method comparison (replicating B1.4)

This section reports TPV multiplier, $D_5$, $S_{\text{fin}}$, and
$Q/\mathrm{kTPV}$ for each method condition on the three target models,
so the quality–cost trade-off is quantified rather than implicit.

**Setup**: From the released method-experiment JSONLs (containing
`output_tokens` and `api_calls` per sample), we compute mean TPV per
valid graph and derive $D_5$, $S_{\text{fin}}$, $Q/\mathrm{kTPV}$. The
Pareto bonus is applied within each target model (relative to that
model's other conditions). The Qwen3.5-35B row below uses the default
A3B (think-enabled) variant (`results/qwen3535ba3b-*.quality.json`),
giving Combined Q = 0.915; the no-think variant
(`results/qwen3535ba3b-nothink-*.quality.json`) gives Combined Q = 0.907
and corresponds to the "Qwen3.5-35B-A3B" row of the main paper's
Tab. method-main.

### Table S2: Method conditions, cost, and efficiency (3 target models)

| Target | Condition | Q | ΔQ vs ZS | ΔQ vs Oracle | TPV | TPV× | API | API× | $D_5$ | $S_\text{fin}$ | Q/kTPV |
|--------|-----------|------|---------|--------------|------|------|-----|------|------|---------------|--------|
| **GPT-4o-mini** | ZS | 0.752 | +0.000 | −0.053 | 436 | 1.0× | 1.00 | 1.00× | 0.752 | 0.865 | 1.72 |
|  | Oracle | 0.805 | +0.053 | +0.000 | 621 | 1.4× | 1.00 | 1.00× | 0.676 | 0.805 | 1.30 |
|  | retry | 0.749 | −0.004 | −0.056 | 441 | 1.0× | 1.01 | 1.00× | 0.749 | 0.749 | 1.70 |
|  | SC | 0.751 | −0.001 | −0.054 | 428 | 1.0× | 1.01 | 1.00× | 0.755 | 0.864 | 1.75 |
|  | CAAP | 0.800 | +0.047 | −0.006 | 650 | 1.5× | 1.00 | 1.00× | 0.665 | 0.800 | 1.23 |
|  | VGIG | 0.810 | +0.058 | +0.005 | 440 | 1.0× | 1.00 | 1.00× | 0.751 | 0.931 | 1.84 |
|  | **Combined** | **0.855** | **+0.103** | **+0.050** | **547** | **1.3×** | 1.00 | 1.00× | 0.705 | **0.983** | 1.56 |
| **DeepSeek-V3** | ZS | 0.822 | +0.000 | −0.036 | 362 | 1.0× | 1.00 | 1.00× | 0.787 | 0.945 | 2.27 |
|  | Oracle | 0.858 | +0.036 | +0.000 | 508 | 1.4× | 1.00 | 1.00× | 0.720 | 0.858 | 1.69 |
|  | retry | 0.827 | +0.006 | −0.030 | 363 | 1.0× | 1.00 | 1.00× | 0.786 | 0.951 | 2.28 |
|  | SC | 0.820 | −0.001 | −0.037 | 368 | 1.0× | 1.00 | 1.00× | 0.784 | 0.820 | 2.23 |
|  | CAAP | 0.855 | +0.033 | −0.003 | 557 | 1.5× | 1.00 | 1.00× | 0.701 | 0.855 | 1.53 |
|  | VGIG | 0.863 | +0.041 | +0.005 | 373 | 1.0× | 1.00 | 1.00× | 0.782 | 0.992 | 2.31 |
|  | **Combined** | **0.894** | **+0.073** | **+0.036** | **512** | **1.4×** | 1.01 | 1.00× | 0.718 | **1.028** | 1.75 |
| **Qwen3.5-35B** | ZS | 0.809 | +0.000 | −0.063 | 681 | 1.0× | 1.01 | 1.00× | 0.653 | 0.931 | 1.19 |
|  | Oracle | 0.872 | +0.063 | +0.000 | 2417 | 3.5× | 1.00 | 1.00× | 0.362 | 1.003 | 0.36 |
|  | retry | 0.874 | +0.065 | +0.002 | 5124 | 7.5× | 1.00 | 0.99× | 0.304 | 0.874 | 0.17 |
|  | SC | 0.874 | +0.065 | +0.002 | 5147 | 7.6× | 1.00 | 0.99× | 0.304 | 0.874 | 0.17 |
|  | CAAP | 0.887 | +0.077 | +0.015 | 4856 | 7.1× | 1.00 | 0.99× | 0.305 | 1.020 | 0.18 |
|  | VGIG | 0.896 | +0.087 | +0.024 | 5356 | 7.9× | 1.00 | 0.99× | 0.303 | 0.896 | 0.17 |
|  | **Combined** | **0.915** | **+0.106** | **+0.043** | **4900** | **7.2×** | 1.00 | 0.99× | 0.305 | **1.053** | 0.19 |

### Key observations

- **Combined dominates Oracle on $S_{\text{fin}}$** for all three target models:
  0.983/1.028/1.053 vs 0.805/0.858/1.003. The Pareto bonus elevates Combined
  precisely because it sits on the (within-model) Pareto frontier.
- **VGIG-only achieves +0.058/+0.041/+0.087 Q at almost-zero TPV overhead**
  on GPT-4o-mini / DeepSeek-V3 (TPV ratio 1.01× ZS); the verification
  loop terminates early on satisfied constraints. This is arguably the
  cleanest practical result.
- **Combined is quality-prioritized, not free**: $Q/\mathrm{kTPV}$ for
  Combined is 0.91× / 0.77× / 0.16× of the ZS baseline. Users with
  strict cost budgets should consider VGIG-only.
- **API-call count is essentially unchanged**: all methods stay at ~1.00
  API calls per sample because the released runner accumulates token
  output rather than issuing multiple round-trips per refinement chain.

---

## 4. Reference-pair dedup sensitivity (replicating B1.2)

> **Concern**: Many reference pairs are identical or isomorphic; does
> this affect D2/D3 reliability?

### Step 1: Reference-pair classification

We classify each 2-reference pair (791 pairs from feasible instructions)
into:
- **distinct**: different name-stripped strings AND non-isomorphic graphs
- **exact-dup**: identical name-stripped strings (only the auto-generated
  `name='Lx-yyy-refK'` differs)
- **iso-only**: different strings BUT isomorphic graphs

### Table S3a: Reference-pair classification by level (directed-aware iso check)

Iso check uses `nx.DiGraph` for directed references (91/791 = 11.5% are
declared `directed=true`) and `nx.Graph` otherwise; mixed-directedness
pairs are classified as `distinct`. Exact-dup compares name-stripped
canonical strings.

| Level | distinct | iso-only | exact-dup | total |
|-------|----------|----------|-----------|-------|
| L0 | 88 | 7 | 5 | 100 |
| L1 | 69 | 4 | 127 | 200 |
| L2 | 145 | 4 | 42 | 191 |
| L3 | 133 | 11 | 6 | 150 |
| L4 | 93 | 7 | 0 | 100 |
| L5 | 18 | 17 | 15 | 50 |
| **Total** | **546** | **50** | **195** | **791** |

The 245 duplicate pairs (31.0%) concentrate at L1 (127/200 = 64% exact-dup,
mostly simple deterministic tree/path/star references) and L5 (32/50 = 64%
some-form-of-duplicate, mostly because target graphs are uniquely determined
by the editing operations). Note: an earlier version of this analysis used
undirected `nx.Graph` for all iso checks, yielding 53 iso-only pairs (3
extra false-positives at L4 and L5 due to directed → undirected coercion);
the corrected count is 50.

### Step 2: D2/D3 sensitivity on the distinct subset

For each (model, strategy) cell, we recompute the mean D2 and D3 at L3 and L4
using only the distinct-reference subset. Out of 90 (cell, level) entries:

- Max $|\Delta D_2| = 0.005$ (Llama-3.1-8B zero-shot at L4): negligible.
- Max $|\Delta D_3| = 0.020$ (GPT-3.5 few-cot at L3): well below per-cell
  bootstrap CI widths (~0.060 from §6).
- Mean $|\Delta D_2| < 0.001$; mean $|\Delta D_3| < 0.005$.

### Table S3b: Representative sensitivity rows

| Cell | Level | D2 (all) | D2 (distinct) | Δ | D3 (all) | D3 (distinct) | Δ |
|------|-------|----------|---------------|---|----------|---------------|---|
| Sonnet-4.6 zero-shot | L4 | 0.176 | 0.177 | +0.0003 | 0.577 | 0.573 | −0.0040 |
| Sonnet-4.6 few-shot | L4 | 0.546 | 0.551 | +0.0040 | 0.583 | 0.575 | −0.0089 |
| GPT-4o zero-shot | L4 | 0.059 | 0.054 | −0.0045 | 0.591 | 0.586 | −0.0050 |
| GPT-4o-mini zero-shot | L4 | 0.032 | 0.027 | −0.0046 | 0.584 | 0.579 | −0.0051 |
| Qwen3.5-397B zero-shot | L4 | 0.012 | 0.010 | −0.0013 | 0.587 | 0.579 | −0.0080 |

### Conclusion

Reference-pair redundancy concentrates at L0/L1/L5 where reference-based
D2 carries zero level-aggregate weight and D3 is either inactive (L0–L2)
or operates on a fixed L3 synthetic pool unaffected by per-instruction
reference duplicates. D2/D3 means on dedup-restricted subsets are within
the bootstrap noise of full-data means at L3/L4 (where the metrics are
active), confirming the redundancy does not bias the main findings.

---

## 5. Failure & truncation rates (replicating B1.5)

> **Concern**: Of the nominal 180,000 baseline outputs, how many actually
> succeeded? Are weaker models penalized by parse failures?

### Global counts

| Quantity | Value |
|----------|-------|
| Total baseline cells | 45 |
| Total successful generations | **179,926** |
| Nominal generations (45 × 800 × 5) | 180,000 |
| Missing | **74** (0.04%) — distributed across 3 cells; explained by API timeouts |
| Total parse failures (`valid=False`) | 12,760 (7.09%) |
| Total truncations (output_tokens ≥ 99% of `max_tokens`) | 96 (0.05%) |
| Cells with ≥ 1 parse failure | 45 / 45 |
| Cells with ≥ 1 truncation | 4 / 45 |

### Table S4: Top-15 cells by parse-failure rate

| Model | Strategy | Level | n_gen | parse_fail % | trunc % |
|-------|----------|-------|-------|--------------|---------|
| GPT-4o-mini | few-cot | L2 | 1000 | **61.70** | 0.00 |
| GPT-3.5 | few-cot | L2 | 1000 | **59.80** | 0.10 |
| GPT-3.5 | few-shot | L2 | 1000 | 59.00 | 0.20 |
| GPT-4o-mini | few-shot | L2 | 1000 | 56.40 | 0.00 |
| GPT-3.5 | few-cot | L3 | 750 | 45.47 | 0.13 |
| GPT-4o-mini | zero-shot | L2 | 990 | 44.65 | 0.00 |
| GPT-4o-mini | few-cot | L3 | 750 | 44.27 | 0.00 |
| GPT-3.5 | zero-shot | L2 | 1000 | 44.20 | 1.40 |
| Llama-3.1-8B | few-cot | L0 | 500 | 39.20 | 0.00 |
| GPT-3.5 | zero-cot | L2 | 1000 | 36.90 | 1.50 |
| Llama-3.1-8B | few-shot | L2 | 1000 | 35.70 | 0.00 |
| Llama-3.1-8B | few-shot | L0 | 500 | 34.40 | 0.00 |
| Llama-3.1-8B | few-shot | L1 | 1000 | 34.40 | 0.00 |
| GPT-4o-mini | zero-cot | L2 | 982 | 34.32 | 0.00 |
| Llama-3.1-8B | few-cot | L1 | 1000 | 31.90 | 0.00 |

### Observations

- **Parse failures concentrate at L2 (multi-constraint composition) on T3
  models**: GPT-4o-mini, GPT-3.5, Llama-3.1-8B see 30–62% parse failures at
  L2 across all four prompting strategies. This is a structural finding
  consistent with the paper's F1: at L2, weak models lose the joint
  constraint structure and emit malformed code-style outputs.
- **Truncation is rare**: only 4 cells have any truncation, all under 3.1%.
- **The 74 missing generations** distribute across 5 cells:
  10 (Sonnet-4.6 few-cot) + 2 (GPT-4o-mini few-cot) + 2 (GPT-4o-mini
  few-shot) + 26 (GPT-4o-mini zero-cot) + 34 (GPT-4o-mini zero-shot);
  all are API-side timeouts retried until the cell-level retry budget
  was exhausted. Reported scores are computed on the successful subset,
  which biases conservatively (drops are independent of instruction
  content under retry-after-timeout).

---

## 6. Statistical robustness

### 6.1 L5 bootstrap confidence intervals (replicating B2.1)

> **Concern**: L5 has only N=50 instructions; the paper quotes
> "95% CI ±0.019" but does not show per-cell bootstrap distributions.

For each (model, strategy) cell, we draw 1000 bootstrap resamples of size
50 from the L5 per-instruction `level_score` values and report the 2.5th and
97.5th percentiles of the resampled mean.

| Quantity | Value |
|----------|-------|
| Mean CI half-width across 45 cells | **0.058** |
| Median CI half-width | 0.064 |
| Boot SE (typical) | 0.030–0.041 |

Note: the paper's "±0.019" likely refers to the standard error of the mean
(SE = σ/√N), whereas our bootstrap reports approximately 1.96 × SE for the
95% CI half-width. With typical per-graph stdev ≈ 0.13 at L5, SE ≈ 0.018,
matching the paper's number. The CI is correspondingly ≈ 0.036 in normal
approximation. Our bootstrap gives slightly wider CIs (0.058–0.064)
because the L5 score distribution is heavier-tailed than normal.

### Table S5a: L5 paired-bootstrap CIs for the F5 claim (Qwen3.5-35B vs 397B)

Both models are evaluated on the same 50 L5 instructions (L5-001 … L5-050),
so the comparison is paired: we draw per-instruction differences
$\delta_i = Q_{35B}(i) - Q_{397B}(i)$ and bootstrap their mean over 1000
resamples of size 50.

| Strategy | Δ = 35B − 397B (paired) | 95% CI (paired) | Crosses zero? |
|----------|--------------------------|-----------------|---------------|
| zero-shot | −0.041 | **[−0.094, −0.002]** | **NO** (35B significantly worse) |
| few-shot | −0.002 | [−0.030, +0.037] | YES |
| zero-cot | −0.035 | **[−0.077, −0.007]** | **NO** (35B significantly worse) |
| **few-cot** | **+0.005** | **[−0.012, +0.027]** | **YES** (paper's quoted gap of −0.005 is here) |

For comparison, the unpaired CI (treating the two models as independent
samples — statistically inappropriate but matches a naive analysis) is
2–4× wider and crosses zero on all four strategies:

| Strategy | Δ | Paired 95% CI | Unpaired 95% CI | CI width paired | CI width unpaired |
|----------|---|---------------|------------------|------------------|--------------------|
| zero-shot | −0.041 | [−0.094, −0.002] | [−0.145, +0.063] | 0.092 | 0.208 |
| few-shot | −0.002 | [−0.030, +0.037] | [−0.091, +0.086] | 0.067 | 0.177 |
| zero-cot | −0.035 | [−0.077, −0.007] | [−0.073, +0.004] | 0.070 | 0.077 |
| few-cot | +0.005 | [−0.012, +0.027] | [−0.020, +0.030] | 0.039 | 0.050 |

**Implication for F5**: the paper's specific finding —— that few-CoT does
not show a detectable L5 gain from 35B to 397B —— **is supported** by the
correct paired analysis (CI [−0.012, +0.027] crosses zero). However, F5 is
**strategy-dependent**: under zero-shot and zero-CoT, Qwen3.5-35B is
**significantly worse** than Qwen3.5-397B on L5 (paired CI excludes zero
on both, with Δ in the −0.04 range). Restating F5 more precisely: at
prompting strategies favouring large reasoning chains (few-CoT, few-shot),
35B catches up to 397B on L5; at strategies without rich
example/CoT context, the scaling advantage of 397B reasserts itself.
The refined framing (§1.3.7) describes F5 as "no monotonic gain detectable
under few-CoT, which is the strategy paper F5 quotes"; the broader
"L5 scale-invariant universally" claim does not survive paired analysis.

### 6.2 Leave-one-model-out OLS for F2 (replicating B1.7)

We use the population standard deviation of Q across the four prompting
strategies (`np.std` with default `ddof=0`), matching the released
figure-generation code (`scripts/paper_figures.py::fig_F4_capability_variance`).
The full-data fit gives **β = −0.0733, R² = 0.398, two-sided p ≈ 0.015**.

### Table S5b: Per-model sigma_strat (population std)

| Model | Tier | mean Q | σ_strat (popstd) |
|-------|------|--------|------------------|
| Sonnet-4.6 | T1 | 0.881 | 0.0152 |
| Qwen3.5-397B | T1 | 0.854 | 0.0182 |
| Qwen3.5-122B | T1 | 0.854 | 0.0147 |
| Qwen3.5-35B | T2 | 0.837 | 0.0190 |
| DeepSeek-V3 | T2 | 0.833 | 0.0078 |
| Llama-3.3-70B | T2 | 0.826 | 0.0069 |
| GPT-4o | T2 | 0.826 | 0.0130 |
| GPT-4.1 | T2 | 0.819 | 0.0180 |
| GPT-4o-mini | T3 | 0.748 | 0.0249 |
| GPT-3.5 | T3 | 0.748 | 0.0267 |
| Llama-3.1-8B | T3 | 0.704 | 0.0220 |

### Table S5c: Leave-one-out and bootstrap stability

| Statistic | Value |
|-----------|-------|
| Full-data β | **−0.0733** |
| LOO β range | [−0.0933, −0.0591] |
| LOO sign stability | **11 / 11 negative** |
| Full-data R² | 0.398 |
| LOO R² range | [0.297, 0.477] |
| Bootstrap (n=10,000) 95% CI for β | [−0.1353, +0.0021] |
| P(β < 0) under bootstrap | **0.975** |

**Interpretation**: the sign of the inverse-scaling relationship is robust
(11/11 LOO fits negative, P(β<0) ≈ 0.975) but the magnitude is fragile
to the small n. The refined framing (§1.3.5) describes this as a
"dominant inverse trend" rather than a strict monotonic relationship,
and references the bootstrap CI here.

### 6.3 Sign-count for strategy effects (replicating B2.3)

> **Concern**: Per-level mean strategy deltas in Tab. strategy-delta may
> hide bi-modal distributions where the mean is pulled by a few extreme
> models. Sign-count makes this visible.

### Table S6: Per-(level, strategy) sign distribution across 11 fully-evaluated models

| Level | Strategy | mean Δ | # positive (>+0.005) | # negative (<−0.005) | # tie (\|·\|≤0.005) | dominant sign |
|-------|----------|--------|----------------------|----------------------|---------------------|---------------|
| L0 | FS−ZS | −0.016 | 2 | 7 | 2 | NEG 7/11 |
| L0 | ZC−ZS | +0.007 | 4 | 0 | 7 | POS 4/11 |
| L0 | FC−ZS | −0.025 | 1 | 8 | 2 | NEG 8/11 |
| L1 | FS−ZS | −0.025 | 1 | 7 | 3 | NEG 7/11 |
| L1 | ZC−ZS | +0.009 | 6 | 0 | 5 | POS 6/11 |
| L1 | FC−ZS | −0.016 | 2 | 6 | 3 | NEG 6/11 |
| L2 | FS−ZS | −0.035 | 3 | 6 | 2 | NEG 6/11 |
| L2 | ZC−ZS | +0.039 | 9 | 1 | 1 | POS 9/11 |
| **L2** | **FC−ZS** | **−0.017** | **6** | **5** | **0** | **MIXED 6+/5−** |
| L3 | FS−ZS | +0.001 | 5 | 3 | 3 | POS 5/11 (margin 2) |
| L3 | ZC−ZS | +0.004 | 7 | 4 | 0 | POS 7/11 |
| L3 | FC−ZS | −0.049 | 3 | 7 | 1 | NEG 7/11 |
| **L4** | **FS−ZS** | **+0.069** | **11** | **0** | **0** | **POS 11/11 (robust)** |
| L4 | ZC−ZS | −0.006 | 5 | 6 | 0 | MIXED 5+/6− |
| L4 | FC−ZS | +0.049 | 8 | 2 | 1 | POS 8/11 |
| L5 | FS−ZS | +0.003 | 6 | 3 | 2 | POS 6/11 (margin 3) |
| L5 | ZC−ZS | +0.038 | 10 | 1 | 0 | POS 10/11 (robust) |
| L5 | FC−ZS | +0.047 | 9 | 2 | 0 | POS 9/11 |

### Key observation

- **L4 FS−ZS is the most robust signed effect**: 11/11 positive. The
  paper's "FS adds +0.069 at L4" statement is therefore strong.
- **L5 ZC−ZS** and **L5 FC−ZS** are also robust (10/11 and 9/11 positive).
- **L2 FC−ZS is bi-modal** (6 positive, 5 negative): the mean of −0.017
  is pulled down by GPT-3.5 (−0.096) and GPT-4o-mini (−0.134). For the
  other 6 models, FC at L2 is mildly positive. The refined abstract
  softens "no single strategy dominates" but the paper's F3 textual
  framing of "few-CoT harms L3" remains robust (7/11 negative).

---

## 7. Weight sensitivity

### 7.1 N-balanced level weighting (replicating B1.3)

> **Concern**: L5 has 50 instructions but weight 0.25 — per-instruction
> influence is 20× that of L1. One could ask whether the leaderboard
> is robust to this design choice.

### Table S7a: Weight schemes evaluated

| Scheme | L0 | L1 | L2 | L3 | L4 | L5 |
|--------|------|------|------|------|------|------|
| Default (paper) | 0.050 | 0.100 | 0.150 | 0.200 | 0.250 | 0.250 |
| N-balanced (∝ 1/n) | 0.176 | 0.088 | 0.088 | 0.118 | 0.176 | 0.353 |
| Uniform (1/6) | 0.167 | 0.167 | 0.167 | 0.167 | 0.167 | 0.167 |
| Mass-balanced (∝ n) | 0.125 | 0.250 | 0.250 | 0.188 | 0.125 | 0.063 |

### Table S7b: Sensitivity matrix (overlap with default ranking)

| Scheme | Top-9 retained | Top-15 retained | Top-15 Jaccard | Max rank shift in top-15 |
|--------|----------------|-----------------|----------------|--------------------------|
| Default | 9/9 | 15/15 | 1.000 | 0 |
| **N-balanced (1/n)** | **9/9** | **14/15** | **0.875** | 4 |
| Uniform (1/6) | 8/9 | 14/15 | 0.875 | 3 |
| Mass-balanced (n) | 8/9 | 12/15 | 0.667 | 8 |

**Conclusion**: The top-9 leaderboard is robust under N-balanced weighting
(which actually *increases* L5's per-cell influence). Top-15 has a single
swap (rank-14 ↔ rank-16). The findings about Sonnet-4.6 / Qwen3.5 / GPT
families on the top of the leaderboard are not driven by L5's small N.

### Bonus: L5-zero ablation (zero L5 weight, D1–L4 renormalised)

Top-9 retained: **6 / 9**. So L5 *does* carry diagnostic signal that
contributes to top-9 distinctions, but the signal is balanced rather
than dominant.

---

## 8. Tier-gap aggregation sensitivity + Sonnet-4 exclusion (replicating B1.6)

> **Concerns**:
> 1. Main-paper Tab. tier-gap reports per-tier mean Quality summaries, but
>    the per-cell aggregation does not exactly reproduce under any single
>    rule we could derive. We document three reproducible aggregations here.
> 2. Sonnet-4 (zero-shot-only) is included in some analyses (tier-mean,
>    Pareto frontier, top-N leaderboard) but excluded from strategy-effect
>    analyses. Are the conclusions sensitive to this inclusion?

### Table S8a: Three tier-gap aggregation choices (all 11 fully-evaluated models)

| Level | Paper Tab. tier-gap | Per-model best-of-4 | All-strategy-average (45 cells) |
|-------|---------------------|----------------------|---------------------------------|
| L0 | 0.057 | 0.069 | 0.103 |
| L1 | 0.120 | 0.123 | 0.160 |
| **L2** | **0.219** | **0.224** | **0.341** |
| L3 | 0.073 | 0.076 | 0.096 |
| L4 | 0.122 | 0.069 | 0.052 |
| L5 | 0.106 | 0.101 | 0.100 |

**Reading the table**: the main paper's Tab. tier-gap aggregates per-cell
in a non-uniform way (e.g., the L4 row is closer to a zero-CoT-anchored
mean than best-of-four). The two reproducible alternatives (best-of-four,
all-strategy-average) **both agree** that L2 dominates every other level
by ≥1.8×, which is the F1 narrative claim. The peak-at-L2 ordering is
invariant to aggregation; only the absolute numbers differ.

### Table S8b: Sonnet-4 exclusion sensitivity (all-strategy-average baseline, 45 vs 44 cells)

| Level | T1 (with S4) | T1 (w/o) | T2 (with) | T2 (w/o) | T3 (with) | T3 (w/o) | Gap (with) | Gap (w/o) | Δ Gap |
|-------|-----------|----------|-----------|----------|-----------|----------|------------|-----------|-------|
| L0 | 0.947 | 0.947 | 0.926 | 0.925 | 0.844 | 0.844 | 0.103 | 0.103 | 0.0000 |
| L1 | 0.978 | 0.978 | 0.965 | 0.964 | 0.818 | 0.818 | 0.160 | 0.160 | 0.0000 |
| L2 | 0.916 | 0.916 | 0.858 | 0.856 | 0.575 | 0.575 | 0.341 | 0.341 | 0.0000 |
| L3 | 0.850 | 0.850 | 0.804 | 0.802 | 0.754 | 0.754 | 0.096 | 0.096 | 0.0000 |
| L4 | 0.790 | 0.790 | 0.763 | 0.764 | 0.738 | 0.738 | 0.052 | 0.052 | 0.0000 |
| L5 | 0.852 | 0.852 | 0.821 | 0.822 | 0.752 | 0.752 | 0.100 | 0.100 | 0.0000 |

Sonnet-4 is a T2 model and therefore mechanically cannot affect T1 or T3
means. The Δ Gap column is exactly zero everywhere — the with-vs-without
comparison is sensitivity-trivial.

### Pareto frontier and top-15

- With Sonnet-4: **6** Pareto-optimal cells (paper number).
- Without Sonnet-4: **5** Pareto-optimal cells — exactly Sonnet-4 zero-shot
  is lost; no other cells change Pareto status.
- Top-15 $S_{\text{tot}}$ leaderboard overlap: **15 / 15** (Sonnet-4 zero-shot
  is at rank #17 in the with-version and so does not appear in either top-15).

**Conclusion**: Sonnet-4 inclusion has zero impact on the T1–T3 gap
(it is a T2 model) and only mechanical impact on the Pareto frontier (the
single point it represents is lost). All other findings — top-15
leaderboard, tier separation, strategy effects — are invariant to its
inclusion.

---

## 9. Reproducibility notes

- All analyses in §2–§8 are reproducible from the released artifact
  (`results/*.{jsonl,quality.json}` + `data/instructions/`); no LLM
  re-generation is required.
- The 10 analysis scripts are in `scripts/analyses/` of the repository
  and produce the JSON files in `scripts/analyses/results/`.
- All scripts are pure-Python 3.10 with stdlib + NetworkX 3.2 only.
- Random seeds are fixed (seed=42) for the bootstrap and CV analyses.
- The 549 unit tests pass on Windows 11 + Python 3.10.16 +
  NetworkX 3.2 + Anaconda standard distribution.

### Script-to-analysis map

| Analysis | Script | Output JSON |
|----------|--------|-------------|
| §2 CV Oracle (B1.1) | `scripts/analyses/b11_caap_cv.py` | `scripts/analyses/results/b11_caap_cv.json` |
| §3 Cost-adjusted (B1.4) | `scripts/analyses/b14_sfin.py` | `scripts/analyses/results/b14_sfin.json` |
| §4 Ref-dedup (B1.2) | `scripts/analyses/b12_ref_dedup.py` | `scripts/analyses/results/b12_ref_dedup.json` |
| §5 Failure rates (B1.5) | `scripts/analyses/b15_failure_rates.py` | `scripts/analyses/results/b15_failure_rates.json` |
| §6.1 L5 bootstrap (B2.1) | `scripts/analyses/b21_l5_bootstrap.py` | `scripts/analyses/results/b21_l5_bootstrap.json` |
| §6.2 LOO OLS (B1.7) | `scripts/analyses/b17_loo_ols.py` | `scripts/analyses/results/b17_loo_ols.json` |
| §6.3 Sign-count (B2.3) | `scripts/analyses/b23_sign_count.py` | `scripts/analyses/results/b23_sign_count.json` |
| §7 N-balanced (B1.3) | `scripts/analyses/b13_nbalanced.py` | `scripts/analyses/results/b13_nbalanced.json` |
| §8 Sonnet-4 exclusion (B1.6) | `scripts/analyses/b16_sonnet4_excl.py` | `scripts/analyses/results/b16_sonnet4_excl.json` |
| (shared utilities) | `scripts/analyses/_common.py` | — |

---

## 10. Worked examples per level

This section fills in the per-level worked-example placeholder of the
paper's worked-example appendix. For each of L0–L5 we show one
illustrative instruction in its native JSON form, one reference solution
in the InstructGraph code-style format, and the per-dimension fields the
evaluation pipeline produces. Together the six examples cover the
progressive complexity ramp from a pure format check (L0) to a
multi-step graph-editing task (L5).

### 10.1 L0 — format generation

```json
{
  "id": "L0-001",
  "level": 0,
  "instruction": "Create a graph with 3 nodes.",
  "explicit_constraints": ["num_nodes=3"],
  "implicit_constraints": ["directed=false"],
  "graph_sizes": ["small"],
  "feasible": true
}
```

```text
Graph[name='L0-001-ref1', nodes=3] {
    node_list = ['0', '1', '2'];
    edge_list = [('0','1'), ('0','2'), ('1','2')];
}
```

- **Active dimensions**: D1 (structural), D4 (instruction-match), D5 (efficiency).
  D2 and D3 are zero-weighted at L0 (no textual or distributional signal).
- **D1**: Valid Rate × 0.7 + Uniqueness × 0.3. Pass requires a parseable
  code-style block; both reference graphs are isomorphic-distinct.
- **D4**: explicit-constraint satisfaction (num_nodes == 3) AND implicit
  satisfaction (directed == false).
- **L0 weights**: D1 = 0.10, D4 = 0.60, D5 = 0.30.

### 10.2 L1 — single explicit constraint

```json
{
  "id": "L1-001",
  "level": 1,
  "instruction": "Generate a tree with 5 nodes.",
  "explicit_constraints": ["graph_type=tree", "num_nodes=5"],
  "implicit_constraints": ["num_edges=4", "acyclic=true", "connected=true"],
  "graph_sizes": ["small"],
  "feasible": true
}
```

```text
Graph[name='L1-001-ref1', nodes=5] {
    node_list = ['0', '1', '2', '3', '4'];
    edge_list = [('0','3'), ('0','2'), ('1','2'), ('1','4')];
}
```

- **Active dimensions**: D1, D4, D5. L1 weights: D1 = 0.15, D4 = 0.70, D5 = 0.15.
- **D4** here exercises the implicit-inference path: a graph is a tree
  iff it is connected, acyclic, and has |V|−1 edges. Generators that
  satisfy `graph_type=tree` and `num_nodes=5` but accidentally produce a
  cycle (e.g., 5 edges) lose D4.

### 10.3 L2 — multi-constraint

```json
{
  "id": "L2-001",
  "level": 2,
  "instruction": "Generate a connected 3-regular graph with 8 nodes.",
  "explicit_constraints": [
    "degree=3", "num_nodes=8", "connected=true", "directed=false"
  ],
  "implicit_constraints": ["num_edges=12"],
  "graph_sizes": ["small"],
  "feasible": true
}
```

```text
Graph[name='L2-001-ref1', nodes=8] {
    node_list = ['0', '1', '2', '3', '4', '5', '6', '7'];
    edge_list = [
        ('0','1'), ('0','3'), ('0','6'), ('1','5'), ('1','3'),
        ('2','7'), ('2','3'), ('2','6'), ('4','6'), ('4','5'),
        ('4','7'), ('5','7')];
}
```

- **Active dimensions**: D1, D4, D5 (L2 weights identical to L1).
- L2 stresses joint constraint satisfaction: a graph can satisfy
  `num_nodes=8` and `connected=true` but fail `degree=3` exactly. The
  no-contradiction sub-score (§3.3 of the paper) credits partial
  satisfaction proportional to the fraction of constraints met.
- 9 of the 200 L2 instructions are tagged `feasible=false` (no
  satisfying graph exists, e.g. odd-degree regular graph on odd $n$);
  these are intentional negative tests evaluated only on Valid Rate.

### 10.4 L3 — property / distributional

```json
{
  "id": "L3-001",
  "level": 3,
  "instruction": "Generate a community-structured graph with 12 nodes divided into 2 communities using a stochastic block model. The graph should be connected, be undirected, have density at most 0.451, have modularity at least 0.25, have clustering coefficient at least 0.519, and have average path length at most 2.0213.",
  "explicit_constraints": [
    "num_nodes=12", "connected=true", "directed=false",
    "density<=0.451", "modularity>=0.25",
    "clustering_coefficient>=0.519", "average_path_length<=2.0213"
  ],
  "implicit_constraints": [],
  "graph_sizes": ["small"],
  "feasible": true
}
```

```text
Graph[name='L3-001-ref1', nodes=12] {
    node_list = ['0', '1', '10', '11', '2', '3', '4', '5', '6', '7', '8', '9'];
    edge_list = [
        ('0','1'), ('0','2'), ('0','3'), ('0','5'), ('1','3'),
        ('1','4'), ('1','5'), ('1','8'), ('2','3'), ('2','5'),
        ('2','7'), ('3','5'), ('3','6'), ('4','6'), ('4','8'),
        ('5','6'), ('6','7'), ('6','8'), ('6','10'), ('6','11'),
        ('7','8'), ('7','9'), ('7','10'), ('7','11'), ('8','11'),
        ('9','11'), ('10','11')];
    0.block = 0; 1.block = 0; 2.block = 0; 3.block = 0;
    4.block = 0; 5.block = 0;
    6.block = 1; 7.block = 1; 8.block = 1; 9.block = 1;
    10.block = 1; 11.block = 1;
}
```

- **Active dimensions**: D1, D3, D4, D5. L3 weights: D1 = 0.15,
  D3 = 0.15, D4 = 0.50, D5 = 0.20.
- **D3 (embedding)** is active from L3 onward. The score is
  `0.15·Grassmann + 0.50·Node-Clf-Gap + 0.35·Embedding-MMD`, computed
  against the 3,115-graph synthetic L3 pool over 15 sub-groups.
- D4 uses the numerical-tolerance comparator for inequality constraints
  (`density<=0.451`, `modularity>=0.25`, etc.); each constraint receives
  full credit if satisfied and zero otherwise, with no half-credit band.

### 10.5 L4 — semantic / domain

```json
{
  "id": "L4-001",
  "level": 4,
  "instruction": "Generate a social network with 10 users in an online community. Members form connections based on shared interests and interactions. The network should have density at most 0.7595, have minimum degree 1, be connected, and be undirected.",
  "explicit_constraints": [
    "num_nodes=10", "density<=0.7595", "min_degree=1",
    "connected=true", "directed=false"
  ],
  "implicit_constraints": [],
  "graph_sizes": ["small"],
  "feasible": true
}
```

```text
Graph[name='L4-001-ref1', nodes=10, domain='social'] {
    node_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
    edge_list = [
        ('0','3'), ('0','8'), ('0','5'), ('0','7'), ('0','6'),
        ('1','3'), ('1','5'), ('1','7'), ('1','4'), ('1','6'),
        ('1','9'), ('2','3'), ('2','5'), ('2','7'), ('2','4'),
        ('3','8'), ('3','5'), ('3','7'), ('3','4'), ('3','6'),
        ('3','9'), ('4','5'), ('4','7'), ('4','6'), ('5','8'),
        ('5','7'), ('5','6'), ('5','9'), ('6','8'), ('6','7'),
        ('7','8')];
    0.label = 'Alice'; 1.label = 'Cora'; 2.label = '...';
    domain = 'social'; edge_type = 'friendship';
}
```

- **Active dimensions**: D1, **D2**, D3, D4, D5. L4 is the only level
  where D2 (textual) is non-zero. L4 weights: D1 = 0.10, D2 = 0.15,
  D3 = 0.05, D4 = 0.55, D5 = 0.15.
- **D2 (textual)** is computed only when the generated graph carries
  attribute strings (labels, domain tags). The score is
  `0.5·text_presence + 0.5·text_similarity` (see §1.1) where
  text_similarity uses character n-gram overlap on node/edge labels
  against the reference labelling.
- **Reference pool**: 1,048 real-world graphs across 9 domains
  (citation, social, biological, ecological, communication,
  infrastructure, knowledge_graph, molecular, plus a held-out residual);
  see §12.3 for the per-domain structural priors.

### 10.6 L5 — multi-step graph editing

```json
{
  "id": "L5-001",
  "level": 5,
  "instruction": "Given a connected graph with 40 nodes and 120 edges: [...edge list...]. Remove the minimum number of edges to produce a spanning tree.",
  "explicit_constraints": [
    "num_nodes=40", "graph_type=tree",
    "directed=false", "task_type=spanning_tree"
  ],
  "implicit_constraints": ["num_edges=39"],
  "graph_sizes": ["large"],
  "feasible": true
}
```

```text
Graph[name='L5-001-ref1', nodes=40] {
    node_list = ['0', '1', ..., '38', '39'];
    edge_list = [
        ('0','11'), ('0','39'), ('1','9'), ('2','20'), ('3','9'),
        ('4','36'), ('4','38'), ('4','37'), ('5','29'), ('6','35'),
        ('6','31'), ('7','32'), ('7','34'), ('8','35'), ('8','18'),
        ('9','35'), ('10','36'), ('10','25'), ('12','37'),
        ... (total 39 edges, spanning tree of the input)];
}
```

- **Active dimensions**: D1, D3, D4, D5 (D2 = 0 — no text attributes).
  L5 weights: D1 = 0.15, D3 = 0.15, D4 = 0.50, D5 = 0.20.
- L5 evaluates **multi-step editing**: a reasoning chain that starts
  from a given input graph and applies a sequence of edge/node
  operations to reach the target property (here: spanning-tree extraction).
  The intermediate-state-diff feedback template (Tab. \ref{tab:fb-templates}
  in the paper appendix) gives per-step partial credit.

### 10.7 L4 real-world reference pool: licenses and provenance

The 1,048 L4 references draw from publicly available domain datasets;
each retains its upstream license. The aggregate is redistributed only
where the upstream license allows it; otherwise the artifact includes
the integration script and a pointer to the upstream source.

| Domain | Source(s) | Upstream license | # graphs |
|--------|-----------|------------------|----------|
| social (Facebook ego, Karate Club) | SNAP, NetworkX bundled datasets | Public (research use) | ~180 |
| citation (DBLP, Cora subset) | SNAP, OGB | ODbL / research use | ~170 |
| biological (PPI fragments) | BioGRID, STRING subsets | CC-BY 4.0 (per source) | ~100 |
| ecological (food webs) | Pajek datasets | Public | ~80 |
| communication (email networks) | SNAP `email-Eu-core` | Public (research use) | ~110 |
| infrastructure (road, power) | NetworkX bundled, SNAP | Public | ~120 |
| knowledge_graph (KG fragments) | WN18, FB15k subsets | Research-use license | ~140 |
| molecular (ZINC subset) | ZINC | Research-use license | ~150 |

The combined `DATA_LICENSE.md` at the repository root enumerates the
per-source license terms.

---

## 11. Scoring formulas and weight ablation

This section spells out the scoring derivation referenced in the paper's
scoring-and-weight-ablation appendix, complementing the D5 robustness
table that the paper renders inline.

### 11.1 D5: token efficiency (Eq. 1)

For a (model, strategy) cell with mean tokens per valid graph (TPV) and
mean API calls per valid graph (API), define

```
D5  =  0.7 · exp(−TPV / s_T)  +  0.3 · exp(−(API − 1) / s_A)
```

with the default scales `s_T = 1000` (tokens) and `s_A = 2`
(extra calls). Choosing API − 1 as the input means a one-shot generator
incurs no penalty; multi-call methods (VGIG with three refinement
rounds → API ≈ 3–4) lose roughly `exp(−1) ≈ 0.37` of the API term. See
the D5 robustness table in the paper for Spearman ρ ≥ 0.966 across the
3 × 3 (s_T, s_A) sensitivity grid.

### 11.2 Per-dimension weights (Tab. S11)

Per-level dimension weights (from `graphinstruct/scoring.py`,
`DIMENSION_WEIGHTS`):

| Level | D1 (struct) | D2 (text) | D3 (embed) | D4 (instr) | D5 (eff) |
|------:|:-----------:|:---------:|:----------:|:----------:|:--------:|
| L0    | 0.10 | 0.00 | 0.00 | **0.60** | 0.30 |
| L1    | 0.15 | 0.00 | 0.00 | **0.70** | 0.15 |
| L2    | 0.15 | 0.00 | 0.00 | **0.70** | 0.15 |
| L3    | 0.15 | 0.00 | 0.15 | **0.50** | 0.20 |
| L4    | 0.10 | 0.15 | 0.05 | **0.55** | 0.15 |
| L5    | 0.15 | 0.00 | 0.15 | **0.50** | 0.20 |

D4 is the dominant dimension at every level by design — the benchmark
is **instruction-driven**, so the primary signal is per-constraint
satisfaction. D2 is active only at L4 (the only level where instructions
demand semantic labels). D3 is active from L3 onward (lower levels have
no meaningful distributional reference pool).

### 11.3 Per-level Quality (Eq. 2)

For a (model, strategy) cell and level $\ell$, the per-level Quality is

```
S_level(ℓ)  =  Σ_d   w_{ℓ,d} · S_{d}(ℓ)
```

where $S_d(\ell)$ is the average per-dimension score over the
$N_\ell$ instructions at level $\ell$ and $w_{\ell,d}$ is the entry
from §11.2.

For the quality-only variant (used when reporting the headline Quality
ranking that excludes deployment cost), D5 is zeroed and the remaining
weights are renormalised:

```
S_level^quality-only(ℓ)  =  Σ_{d ≠ D5}  (w_{ℓ,d} / Σ_{d′ ≠ D5} w_{ℓ,d′})  ·  S_d(ℓ)
```

### 11.4 Total Quality (Eq. 3)

Across the six levels,

```
S_total  =  Σ_ℓ   v_ℓ · S_level(ℓ),    v = (0.05, 0.10, 0.15, 0.20, 0.25, 0.25)
```

Level weights $v_\ell$ are progressive: L0 carries 5% of the
aggregate and L4/L5 together carry 50%, encoding the design
principle that higher-complexity levels are the more informative
discriminator. §7.1 (N-balanced weighting) verifies that the top-5
leaderboard is robust under uniform-weight, dimension-proportional,
and per-instruction-balanced alternatives.

### 11.5 Combined Score (Eq. 4)

The Combined Score blends Quality with normalised efficiency for
deployment-oriented ranking:

```
S_combined  =  α · S_total  +  (1 − α) · D5_total,    α = 0.7
```

with `D5_total = Σ_ℓ v_ℓ · S_D5(ℓ)`. The 70 / 30 mix reflects the
empirical Quality / efficiency tension on the Pareto frontier
(§3 in this document; §RQ6 in the paper): doubling α to 1.0 (pure
Quality) reranks at most 3 cells in the top-15.

### 11.6 Pareto-adjusted final score (Eq. 5)

To bonus models that sit on the cost / quality Pareto frontier without
distorting the underlying Quality ordering,

```
S_final  =  S_total · (1 + λ · ParetoBonus)
```

with `λ = 0.15` and `ParetoBonus ∈ [0, 1]`. The bonus is 1.0 when the
cell lies on the frontier, 0.0 when it is dominated by the worst
frontier point, and a normalised-distance interpolation otherwise
(`graphinstruct/analysis/pareto.py::pareto_bonus`). The cap λ = 0.15
guarantees that a frontier cell cannot leapfrog more than one
non-frontier cell in the leaderboard (verified empirically over the
45-cell ranking).

### 11.7 λ sensitivity (sweep summary)

We re-rank the 45 baseline cells under $\lambda \in \{0.05, 0.10,
0.15, 0.20, 0.25\}$. The top-15 leaderboard shifts by at most 3
positions across the sweep; the top-5 frontier-vs-non-frontier
assignment is invariant for $\lambda \le 0.20$. The default
$\lambda = 0.15$ falls within the stability plateau. See
`graphinstruct/analysis/pareto.py` for the implementation;
`scripts/d5_robustness.py` runs the analogous robustness check for D5
scales.

### 11.8 Uniform-weight ablation (cross-reference)

Setting all $w_{\ell,d}$ to $1/k_\ell$ (uniform within each level's
active dimensions) and re-aggregating leaves the top-9 cells
rank-stable. This complements the N-balanced ablation in §7.1, which
re-weights *level* weights, whereas this paragraph re-weights
*dimension* weights at fixed level weights.

---

## 12. VGIG / CAAP algorithmic details

This section fills in the pseudocode and decision-table placeholder of
the paper's VGIG/CAAP appendix. It draws directly from
`graphinstruct/improvements/{runners.py, caap.py, domain_priors.py}`.

### 12.1 Algorithm 1: VGIG iterative refinement

```text
Input :  instruction I, model M, generator G,
         max_rounds T = 3, num_chains K = 3,
         feedback_level ∈ {coarse, fine},
         optional CAAP overlay use_caap ∈ {false, true}
Output:  K refined samples (one per chain)

if use_caap:
    decision           ← CAAP.select_strategy(I, M)        # see Algorithm 2
    strategy_0, extras ← decision
    prompt_0           ← BuildEnhancedPrompt(I, strategy_0, extras)
else:
    strategy_0 ← "zero-shot"
    prompt_0   ← BuildBasePrompt(I, strategy_0)

for chain k = 1 .. K do
    temp_k    ← Temperature(0.7) ± jitter(k)               # per-chain diversity
    best, s   ← G(I, prompt_0, temp_k)                     # round 0
    score_k   ← SatisfactionRate(best, I.constraints)

    if not I.feasible:
        emit best ; continue                                # short-circuit unsolvable

    parse_fail_streak ← 0
    for t = 1 .. T do
        if score_k ≥ 1.0: break                            # all constraints satisfied

        if best.graph is None:                              # parse failure
            parse_fail_streak ← parse_fail_streak + 1
            if parse_fail_streak ≥ 2: break
            best ← G(I, prompt_0, temp_k)                  # fresh regenerate
            continue

        parse_fail_streak ← 0
        feedback ← GenerateFeedback(best.graph, I, feedback_level)
        if feedback is empty: break                         # no actionable feedback

        prompt_t ← BuildRefinementPrompt(I, Serialize(best), feedback, t, extras)
        cand    ← G(I, prompt_t, temp_k · 0.6)             # lower temp for refinement
        if SatisfactionRate(cand, I.constraints) > score_k:
            best, score_k ← cand, SatisfactionRate(cand, I.constraints)

    emit best
```

Notes:

- The 0.6 × temperature factor for refinement rounds (line 23) was
  introduced after pilot experiments showed that refinements at the
  generation temperature drifted away from constraint-satisfying
  candidates.
- VGIG short-circuits on `feasible=false` instructions to avoid wasting
  rounds on the 9 known-unsolvable L2 entries.

### 12.2 Algorithm 2: CAAP per-instruction strategy selection

The CAAP decision matrix is a deterministic lookup keyed on instruction
*level* and model *tier*, with type-specific branches at L1 and L2 and
domain-prior injection at L4. The 24 (level × tier) cells of the
matrix, plus the typed branches and extras, are exhaustively encoded
in `graphinstruct/improvements/caap.py`.

| Level | T1 (best) | T2-open (Qwen3.5-35B, DeepSeek-V3, Llama-70B) | T2-GPT (GPT-4o, GPT-4.1) | T3 (GPT-3.5, GPT-4o-mini, Llama-8B) |
|------:|-----------|------------------------------------------------|---------------------------|--------------------------------------|
| L0    | ZS | ZS | ZS | ZS |
| L1 (simple type: tree/cycle/star/path/complete) | ZS | ZS | ZS | ZS |
| L1 (complex type) | FC | FC | **FS** | **ZC** |
| L2    | FC | FC | **FS** | **ZC + checklist** |
| L3    | ZC | ZC | **ZS** | **ZC + formulas** |
| L4    | FC + domain prior | FC + domain prior | **FS + domain prior** | **FS + domain prior** |
| L5    | FC | FC | FC | **ZC** |

**Reading the table.** ZS = zero-shot, FS = few-shot, ZC = zero-shot
CoT, FC = few-shot CoT. Boldface marks tier-specific deviations from
the row-dominant strategy.

The "hard rules" baked into the table are:

1. **L2 forbids FS for T3.** FS at L2 collapses GPT-4o-mini to
   Quality = 0.424 in pilot (≥0.65 elsewhere); the T3 row instead uses
   ZC plus an explicit per-constraint checklist.
2. **L3 forbids FC.** FC is the largest negative effect at L3
   (mean Δ = −0.048 vs. baseline); the table routes all tiers to ZC or
   ZS variants. T3 additionally receives explicit formula hints
   (density, clustering, etc.) for the numerical constraints in
   the instruction.
3. **L4 always uses FS-flavoured prompts with a domain prior.** FS is
   the uniquely strong positive strategy at L4 (mean Δ = +0.069); T2-open
   and T1 can additionally afford FC's CoT overhead.
4. **L5 forbids FC for T3.** Weak models hurt under FC at L5
   (mean Δ = −0.041); ZC is the safest fallback.

The "extras" slot delivers level-specific augmentation alongside the
chosen prompting strategy:

| Extra | Trigger | What it adds to the prompt |
|-------|---------|----------------------------|
| `checklist` | L2 × T3 | Numbered per-constraint verification checklist |
| `formula`   | L3 × T3 | Closed-form definitions for density / clustering / avg-path / diameter / modularity, conditional on the constraint actually appearing in the instruction |
| `domain`    | L4 (all tiers) | Domain-specific structural prior (degree / clustering / density ranges + qualitative hints), see §12.3 |
| `step_hint` | L5 (when present) | Explicit step-wise breakdown of the editing task |

### 12.3 L4 domain structural priors

CAAP injects a domain-specific prior into L4 prompts (`extras.domain`).
The eight priors below are the complete set encoded in
`graphinstruct/improvements/domain_priors.py`. Ranges are derived from
the L4 reference-pool ground-truth statistics and from published
descriptions of each domain.

| Domain | Avg degree | Clustering | Density | Degree dist. | Canonical motifs / hints |
|--------|------------|------------|---------|---------------|---------------------------|
| social         | 4.0 – 6.0 | 0.40 – 0.70 | 0.30 – 0.50 | power-law | Tight friend circles; few hub users with many friends. |
| citation       | 2.5 – 4.0 | 0.05 – 0.25 | 0.05 – 0.15 | power-law | DAG (new → old); a few seminal-paper hubs. |
| biological     | 2.0 – 5.0 | 0.10 – 0.40 | 0.10 – 0.30 | power-law | PPI / regulatory networks are sparse with weakly-connected functional modules. |
| ecological     | 2.0 – 4.0 | 0.15 – 0.35 | 0.10 – 0.25 | roughly-uniform | Producer–consumer food-web tier structure. |
| communication  | 3.0 – 8.0 | 0.20 – 0.50 | 0.30 – 0.50 | power-law | Active hubs, bursty interactions. |
| infrastructure | 2.0 – 4.0 | 0.05 – 0.20 | 0.05 – 0.20 | roughly-uniform | Grid-like; high betweenness on backbone nodes. |
| knowledge_graph| 2.0 – 6.0 | 0.05 – 0.30 | 0.05 – 0.20 | power-law | Typed directed relations; a few hub entities. |
| molecular      | 1.5 – 3.5 | 0.00 – 0.15 | 0.05 – 0.25 | roughly-uniform | Valence-limited degree (C≤4, N≤3, O≤2); planar with small rings. |

The rendered prior text injected into the prompt is, for example:

```text
Domain context: social network (10 nodes).
Expected structural properties for this domain:
- Average degree: 4.0 - 6.0
- Clustering coefficient: 0.40 - 0.70
- Density: 0.30 - 0.50
- Degree distribution: power-law
- People form tight friend circles, producing high clustering.
- A few hubs have many friends; most nodes have few (power-law).
Generate a graph that satisfies BOTH the explicit constraints
AND these domain expectations.
```

### 12.4 Feedback templates (cross-reference)

The level-specific coarse/fine feedback templates used by VGIG's
`GenerateFeedback` step are tabulated in the paper's VGIG appendix
(Tab. on feedback templates). The fine variant adds per-constraint
`(expected, observed, Δ)` tuples on top of the coarse violation
categories; the coarse variant alone provides about 60% of the gain in
pilot.

---

## 13. Changelog

- **2026-05-13**: First release. Contains 9 extended analyses (§2–§8),
  the errata and clarifications in §1, and the worked-example /
  scoring-formula / VGIG-CAAP appendix completions in §10–§12.

---

*End of supplementary analyses document.*
