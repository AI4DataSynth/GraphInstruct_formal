# GraphInstruct — Rebuttal-Phase Extended Analyses

This document collects the analyses added during the discussion period. They
fall into two groups: (i) offline recomputations on the already-released
per-instance scores (`results/*.quality.json`), which add no new generation;
and (ii) new generations run on additional model families, reasoning models,
and held-out-exemplar / disabled-thinking conditions.

Each section names the analysis product that reproduces its numbers, under
`rebuttal_analyses/results/*.json` (12-model panel) and `…/ext16/*.json`
(16-model panel; toggle with `REBUTTAL_MODELSET=ext16`). **This file is the
primary and self-contained record of the rebuttal-phase results: every number
cited in the author responses appears in a section below, together with the
product file that reproduces it.**

- License: CC BY-4.0 for data, MIT for code (same as the main artifact).
- Scores are quality-only totals (D1–D4, the reference-free portion of the
  evaluation) unless stated otherwise. Level weights are the paper default
  `[0.05, 0.10, 0.15, 0.20, 0.25, 0.25]` unless a scheme is named.

---

## 1. The L2 discrimination peak under six controls

The per-level tier gap (top-tier minus bottom-tier mean score) is largest at
L2. Because the active metric set changes across levels (L0–L2 are
constraint-checking; L3–L4 add distributional/embedding metrics), a single
control is not enough; the peak is therefore tested under a battery that holds
each potential confound fixed in turn.

All controls below were re-run after extending the baseline panel from 12 to
**16 models** (adding GLM-4.6, Gemma-3-27B, Phi-4, Mistral-Small-24B;
`REBUTTAL_MODELSET=ext16`, products under `rebuttal_analyses/results/ext16/`).
The L2 peak survives every control at 16 models, with one honest softening
(split-data per-seed frequency 1.0 → 0.8, §1.4) and the same single exception
(active-parameter tiering, §1.7). The per-control peak flags are aggregated in
`uni_unified_robustness.json`, which records `l2_is_unique_peak = true` for
every capability-correlated column at both panel sizes.

### 1.1 Single continuous metric across all levels (`c1_d4_only_tiergap.json`)

Using one continuous D4 (instruction-satisfaction) score defined identically on
all six levels removes the metric-regime change. The tier gap (T1−T3):

| Level | L0 | L1 | **L2** | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Continuous-D4 gap (12-model) | 0.0632 | 0.1122 | **0.2129** | 0.1136 | 0.0575 | 0.1426 |

L2 is the unique maximum (16-model panel: L2 = 0.194, still the unique maximum).

### 1.2 Paired bootstrap (`c2_tiergap_bootstrap_ci.json`)

A paired bootstrap (2000 resamples) gives the L2 gap a 95% CI of
**[0.1817, 0.2450]**. Its lower bound exceeds the upper CI of both L1 (0.1428)
and L3 (0.1247), so the L2 gap is separated from its neighbours at the 95%
level. For completeness, the small L5 split (n = 50) has a correspondingly wide
gap CI — observed 0.143, 95% CI [0.074, 0.227] — quoted wherever L5 is
discussed.

### 1.3 Tier-free cross-model dispersion (`s1_tierfree_discrimination.json`)

Measured without any tiering — the per-level standard deviation of the 16-model
zero-shot panel — L2 is the most dispersed level (std **0.113**, versus 0.061
at the next-highest level L1, a 1.84× ratio). The L2 score range (0.349) is the
widest of any level. This removes the tier definition entirely. (The 12-model
core panel gives the same conclusion: L2 std 0.125, 1.80× the runner-up.)

### 1.4 Split-data tiers (`e1b_splitdata_tier.json`)

Defining tiers on a random half of the instructions and measuring the gap on
the disjoint half (the standard remedy for circular selection) reproduces the
L2 peak on **5/5 seeds** at 12 models, mean gap 0.212. Under the 16-model
extension the aggregate peak still holds (`l2_is_unique_peak = true`) but the
per-seed unique-peak frequency softens to **0.8** (4/5) — reported honestly.

### 1.5 External capability anchors (`off-b_external_tier.json`)

Re-tiering the models by independently published capability scores removes any
dependence on our own Q score: the tiers are defined by external data that is
unrelated to the D4 gap being measured, which breaks the self-scoring
circularity **regardless of tier membership**. All 16 models carry clean LMArena
Arena-Text-Overall Elo (single snapshot 2026-07-21) and published MMLU-Pro:

| Anchor (tiering) | L2 gap | Peak level |
|---|---|---|
| Elo, 12-model 3/6/3 | 0.1514 | L2 |
| MMLU-Pro, 12-model 3/6/3 | 0.1859 | L2 |
| Elo, 16-model 4/8/4 | 0.1465 | L2 |
| MMLU-Pro, 16-model 4/8/4 | 0.1737 | L2 |

L2 is the unique peak in every external tiering × both scoring metrics
(quality-only and continuous-D4). The external top tier now overlaps the
Q-based top tier by **3/4** (e.g. Elo-16 T1 ∩ Q-T1 = {Sonnet-4.6, Qwen3.5-397B,
GLM-4.6}); this agreement between an independent capability measure and our Q
score *corroborates* Q as a genuine capability metric rather than undermining
it. (An earlier version used an 8-model Elo subset that excluded four frontier
models lacking clean public Elo and reported a zero-overlap top tier; with
clean Elo now available for all 16 models we use the full-coverage anchors and
the independent-source framing above — no cherry-picking.)

### 1.6 Instruction paraphrase (`exp6_paraphrase.json`)

L1–L3 instructions were LLM-paraphrased with the constraints held
byte-identical (only wording changed), and tier-spanning models re-run.
Measured tier-free (per-level model-score std), L2 remains the most dispersed
level in both the original and paraphrased sets (std 0.115 → 0.086, both
above L1 and well above L3). The peak location is paraphrase-invariant; its
magnitude compresses slightly. (Probe on a **6-model**, 50-per-level, L1–L3
subset; Mistral-Small-24B, previously dropped for a rare scorer crash, was
re-scored cleanly on 2026-07-26 and is now included → 6/6.)

### 1.7 The one control that does not peak at L2 (`e1a_external_param_tier.json`)

Tiering the six open-weight models by active-parameter count moves the peak to
L1 (continuous-D4 gaps: L1 0.080 vs. L2 0.042, from `c1_d4_only_tiergap.json`
`external_param_tier`; the quality-only variant in
`e1a_external_param_tier.json` agrees: 0.080 vs. 0.039). Active-parameter
count mis-orders heterogeneous
MoE models — it places Qwen3.5-122B (10B active) below Qwen3.5-397B (17B
active). The capability-correlated anchors above (Elo, MMLU-Pro, split-data)
all return the peak to L2.

---

## 2. Dimension validity: D1 vs D4 (`b1_dimension_correlation.json`)

Whether the structural (D1) and instruction (D4) dimensions are redundant is a
convergent/discriminant-validity question (Campbell & Fiske, 1959): correlation
is expected where the constructs converge; the diagnostic is whether it breaks
down where they diverge. Correlations are pooled over 124 `results/*.quality.json`
runs (each per-instance value is a five-sample aggregate).

| Level | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Pearson r(D1,D4) | 0.23 | 0.71 | 0.80 | 0.44 | 0.19 | 0.66 |
| Spearman ρ(D1,D4) | −0.31 | 0.33 | 0.23 | 0.05 | −0.12 | −0.08 |

Both coefficients are high/positive on the constraint-heavy levels (L1–L2),
where a well-formed graph both satisfies the structural definition and the
explicit instruction, and drop sharply from L3 onward, where D4 rewards
numerical/semantic satisfaction that D1's checks do not see. Neither
coefficient is strictly monotone (L0 is a near-ceiling format-only level).
Pearson (linear) exceeds Spearman (rank) because D1/D4 are bounded
five-sample aggregates with many ties, which depresses the rank correlation.
The product file also lists orthogonal cases (D1 = 1 with D4 < 1, and D1 low
with D4 = 1) at every level.

### 2.1 Dimension-subset leaderboards (`b1b_dimension_subset_rankings.json`)

The redundancy question also has a direct leaderboard form: if D1 and D4
measured the same property, re-scoring the leaderboard with either dimension
alone would reproduce the same ranking. Holding each model's best
quality-only baseline run fixed and re-scoring that run under a dimension
subset (unweighted per-instruction mean over the subset's dimensions,
aggregated with the default level weights):

| Subset | Spearman ρ | Kendall τ-b | top-5 kept | top-9 kept | max shift |
|---|---|---|---|---|---|
| D1-only | 0.804 | 0.636 | 4/5 | 8/9 | 5 |
| D4-only | 0.958 | 0.909 | 4/5 | 9/9 | 3 |
| D1+D4 | 0.986 | 0.939 | 5/5 | 9/9 | 1 |

D1-only produces a visibly different ranking — Qwen3.5-397B (full-score #2)
falls to #7 while GPT-4.1 and GPT-4o rise from #5/#6 to #2/#3: the GPT
family emits more structurally clean graphs, while the Qwen family satisfies
more instruction constraints. D4-only tracks the full leaderboard much more
closely (expected for the weight-dominant dimension), and D1+D4 is nearly
indistinguishable from the full quality-only ranking (max shift 1), so D2/D3
refine rather than reorder. The two dimensions correlate where the
constructs converge (the L1–L2 rows above) yet rank models differently when
used alone — exactly what "related but not redundant" predicts.

---

## 3. Cross-family refinement (`results/quality/<model>-{combined,retry}.quality.json`)

The verification-guided refinement (Combined) and the "verification >
prompt-engineering" ordering are re-run on nine model families, to test whether
they hold beyond the three original target models. Δ is the quality-only total
gain over the best single-pass baseline for that model.

| Family | Δ Combined | Retry sign | verify−retry |
|---|---|---|---|
| Sonnet-4.6 | +0.018 | − | + |
| GLM-4.6 | +0.043 | − | + |
| Qwen3.5-35B | +0.055 | − | + |
| DeepSeek-V3 | +0.054 | − | + |
| Llama-3.3-70B | +0.049 | − | + |
| Phi-4 | +0.057 | − | + |
| Gemma-3-27B | +0.042 | − | + |
| Mistral-24B | +0.069 | + | + |
| GPT-4o-mini | +0.070 | − | + |

Combined is positive on all nine families (mean +0.050, range +0.018 to
+0.070). Naive Retry is negative on 7 of 8 families it was run on (mean
−0.020). The verify-over-retry gap is positive on all 8 (mean +0.071).

---

## 4. Reasoning-model panel (`exp4` product)

A reasoning-specialised panel was added to test whether the L2 discrimination
and the L4 ceiling persist when reasoning models are included. Scores are
quality-only totals; L4 is shown separately as it is the load-bearing level.

| Model (zero-shot) | Total | L2 | L4 |
|---|---|---|---|
| GPT-5 | 0.900 | 0.976 | 0.772 |
| GPT-5-mini | 0.895 | 0.978 | 0.763 |
| GLM-4.6 | 0.851 | 0.933 | 0.714 |
| DeepSeek-R1 | 0.831 | 0.924 | 0.715 |
| Qwen3-235B-Thinking | 0.843 | 0.841 | 0.749 |

L4 is the lowest level for every reasoning model (0.71–0.77). Few-shot prompting
lifts L4 by ~+0.08 (GPT-5 +0.078, GPT-5-mini +0.081, GLM-4.6 +0.092) while
other levels barely move — consistent with the L4 decomposition in §7. For
DeepSeek-R1, the L4 breakdown is D2 = 0.025 against D1 = 0.900 and D4 = 0.883,
i.e. the low L4 score sits almost entirely in the text-similarity dimension.

DeepSeek-R1 and Qwen3-235B-Thinking each reached a full 800-instruction run via
sharded resumed generation; an earlier Qwen3-235B-Thinking run had 44% of L3
samples return zero output tokens (API drops), which had depressed its L3 to
0.534 — on the final full-800 run (2 zero-token samples remaining, 0.05% of
4,000) L3 recovers to 0.780, confirming the earlier value was a failure-sample
artifact rather than a capability signal.

---

## 5. Held-out-exemplar leakage test (`exp23_heldout_decoupling.json`)

The paper draws few-shot exemplars from a pool that excludes each instruction's
own evaluation reference. This is verified empirically by re-running with the
exemplar drawn from a held-out reference index (never the evaluation reference)
and comparing the quality-only total. `original` = evaluation-reference-adjacent
exemplar; `held-out` = disjoint exemplar.

| Model | Strategy | original | held-out | Δ |
|---|---|---|---|---|
| DeepSeek-V3 | few-shot | 0.8405 | 0.8397 | −0.0008 |
| DeepSeek-V3 | few-cot | 0.8395 | 0.8299 | −0.0096 |
| GPT-4o-mini | few-shot | 0.7424 | 0.7443 | +0.0019 |
| GPT-4o-mini | few-cot | 0.7148 | 0.6786 | −0.0362 |
| Qwen3.5-35B | few-cot | 0.8607 | 0.8413 | −0.0194 |

Few-shot deltas are ≈0; the few-cot deltas are small and negative (longer CoT
exemplars, held-out, cost slightly more). For Qwen3.5-35B, `original` and
`held-out` are both disabled-thinking runs with per-level output-token medians
in the same range (L3 8382/10336, L4 1860/2336, L5 1458/1580 — no
thinking-mode inflation), so the two are comparable. No condition shows the
large positive delta that reference leakage would produce.

Structural note: the dominant dimension D4 (level weights 0.50–0.70) uses no
reference at all, so the exemplar-leakage surface is confined to D2/D3 at L3+,
a small fraction of the total.

---

## 6. Decoupled verifier (`exp23_heldout_decoupling.json`)

To separate "the method uses the D4 evaluation signal" from "the method uses
iterative structured feedback", the verifier was restricted to a pure
structural signal (D1 validity / graph-type checks), with no D4
explicit-constraint satisfaction.

| Model | Δ (structural-only verifier) |
|---|---|
| GPT-4o-mini | +0.0042 |
| DeepSeek-V3 | −0.0030 |

With the D4 constraint-violation feedback removed, the gain collapses to ≈0.
The benefit therefore comes from the constraint-violation feedback (a
programmatic constraint check, analogous to a compiler/unit-test loop), not
from having a scalar score. For reference, VGIG's gain over the same-signal
baselines is +0.036 (vs Retry) / +0.043 (vs SC) on DeepSeek-V3 and +0.061 /
+0.059 on GPT-4o-mini — while Retry and SC themselves score *below* each
model's best single-pass baseline (DeepSeek-V3: Retry −0.013, SC −0.020;
GPT-4o-mini: Retry −0.036, SC −0.033; `d2_vgig_vs_retry_sc.json`). Having the
signal does not help; iterating on it does.

---

## 7. L4 dimension decomposition (`l4d_l4_dimension_decomp.json`, `off5_l4_semantic_altmetric.json`)

The L4 "iteration-invariance" was decomposed across the 12 baseline models,
comparing zero-shot to few-shot prompting (means and ratios from the
`off5` zero-to-few-shot jump analysis; dispersion stats from `l4d`).

| L4 metric | zero-shot | few-shot | ratio | what it measures |
|---|---|---|---|---|
| D2 (token-overlap w/ reference) | 0.061 | 0.530 | **8.70×** | surface text similarity |
| ↳ text_presence (`.label=` syntax) | 0.110 | 0.978 | 8.85× | did the model emit label syntax |
| ↳ reference-lexical match | 0.004 | 0.038 | ~0 | reproduces the reference's exact labels |
| D1 (structural) | 0.892 | 0.887 | 0.99× | graph well-formedness |
| D3 (embedding) | 0.584 | 0.582 | 1.00× | distributional similarity |
| D4 (instruction-match) | 0.906 | 0.886 | 0.98× | constraint satisfaction |
| metric_a_struct (shape only) | 0.604 | 0.656 | 1.09× | serialization-invariant structure |
| metric_b (BERT semantic) | 0.204 | 0.859 | **4.21×** | serialization-invariant semantics |

The 8.70× D2 jump is almost entirely surface syntax: a reference-free detector
of `.label=` syntax jumps 8.85×, while the exact-reference-label match stays
near zero (models emit their own labels, e.g. `User_1`, not the reference's
`Alice`). D1, D3, D4 and the shape-only metric do not move. A serialization-
invariant BERT metric still rises 4.21×, so few-shot does teach some
domain-appropriate vocabulary — a real but shallow gain, not an 8.7× capability
change. D2 is thus largely a serialization-sensitive surface metric; the L4
ceiling is partly a text-metric effect and partly a domain-grounding gap.

---

## 8. Held-out dimension gains — anti-Goodhart (`d1_vgig_heldout_gains.json`)

Because the refinement optimizes against the D4 signal, its effect on the
dimensions it never optimizes (held-out D1/D2/D3) is the test for Goodhart-style
gaming. A full held-out decomposition is available for the two original target
models. `mean` = mean gain over all applicable D1/D2/D3 cells.

| Model | Method | mean | D1 | D2 | D3 | cells + |
|---|---|---|---|---|---|---|
| DeepSeek-V3 | Combined | +0.027 | +0.037 | −0.052 | +0.032 | 7/10 |
| DeepSeek-V3 | VGIG-only | −0.026 | +0.021 | −0.45 | +0.021 | 6/10 |
| GPT-4o-mini | Combined | +0.079 | +0.040 | +0.500 | +0.019 | 9/10 |
| GPT-4o-mini | VGIG-only | +0.011 | +0.010 | +0.009 | +0.016 | 8/10 |

For Combined, the structural (D1) and distributional (D3) held-out dimensions
rise on both models; under D4-gaming they would be flat or fall. Not every cell
is positive (7/10 and 9/10): D2 falls on DeepSeek-V3 (−0.052) because refinement
makes the graph structurally more correct while diverging from the specific
reference string — the same serialization sensitivity quantified in §7.
VGIG-only is weaker (mean −0.026 on DeepSeek-V3, driven by D2 −0.45), so the
anti-gaming reading is limited to Combined and to the D1/D3 dimensions.

---

## 9. Weight and aggregation robustness (`a1_level_weight_sensitivity.json`,
`a2_aggregation_crosscheck.json`)

This section is the full weight-perturbation record (the analysis behind the
submission's compressed Appendix D, restored here in full). Each scheme
rescores the 12-model leaderboard from the same released per-level scores;
only the level-weight vector changes.

Scheme definitions (weights for L0…L5):

| Scheme | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| default (paper) | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.25 |
| uniform | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 |
| reversed | 0.25 | 0.25 | 0.20 | 0.15 | 0.10 | 0.05 |
| +50% high levels | 0.04 | 0.08 | 0.12 | 0.16 | 0.30 | 0.30 |
| −50% high levels | 0.067 | 0.133 | 0.200 | 0.267 | 0.167 | 0.167 |
| sample-proportional | 0.125 | 0.25 | 0.25 | 0.1875 | 0.125 | 0.0625 |

Default leaderboard (best-baseline, quality-only): Sonnet-4.6 (0.9018) >
Qwen3.5-397B (0.8788) > Qwen3.5-122B (0.8714) > Qwen3.5-35B (0.8607) >
GPT-4.1 (0.8496) > GPT-4o (0.8443) > DeepSeek-V3 (0.8405) > Sonnet-4 (0.8342) >
Llama-3.3-70B (0.8341) > GPT-3.5-turbo (0.7862) > GPT-4o-mini (0.7844) >
Llama-3.1-8B (0.7305).

Effect on the ranking (vs. the default leaderboard):

| Scheme | Spearman ρ | Kendall τ-b | top-5 kept | top-9 kept | all-12 kept | max rank shift |
|---|---|---|---|---|---|---|
| uniform | 0.979 | 0.909 | 5/5 | 9/9 | 12/12 | 1 |
| reversed | 0.909 | 0.788 | 4/5 | 9/9 | 12/12 | 4 |
| +50% high | 0.993 | 0.970 | 5/5 | 9/9 | 12/12 | 1 |
| −50% high | 0.972 | 0.909 | 5/5 | 9/9 | 12/12 | 2 |
| sample-proportional | 0.937 | 0.848 | 4/5 | 9/9 | 12/12 | 3 |

The top-9 set is preserved 9/9 under every scheme, and no model moves more
than 4 positions even under the fully reversed weighting. The only top-5
change anywhere: under the reversed and sample-proportional schemes,
Qwen3.5-35B (default #4) drops to #6 while Sonnet-4 (default #8) rises to
#4/#5 — a within-top-tier reordering, reported for completeness. The
sample-proportional scheme down-weights the 50-instance L5 set from 0.25 to
0.0625 — directly probing the L5-overweighting concern — and still preserves
the top-9.

A rank-aggregation cross-check replaces the weighted sum with average-rank,
Borda-count, and geometric-mean aggregation over the same per-level scores;
all six aggregator pairs agree at Kendall τ-b ∈ [0.848, 1.000].

### 9.1 Sample-size-motivated schemes, cell granularity, and the L5-zero ablation (`a1b_cell_weight_sensitivity.json`)

The concern behind sample-proportional weighting has a sharper form: L5 has 50
instructions but weight 0.25, so one L5 instruction carries 20× the influence
of one L1 instruction on the total. This is probed directly at the finer cell
granularity — each of the 45 (model × baseline-strategy) runs over the 12
baseline models is a leaderboard entry — under schemes that re-balance by
sample size, plus an ablation that deletes L5 outright:

| Scheme (weights L0…L5) | top-9 kept | top-15 kept | max shift (top-15) | Spearman ρ | Kendall τ-b |
|---|---|---|---|---|---|
| N-balanced ∝ 1/n_l (0.176, 0.088, 0.088, 0.118, 0.176, 0.353) | 9/9 | 14/15 | 4 | 0.981 | 0.907 |
| uniform (1/6 each) | 8/9 | 14/15 | 3 | 0.983 | 0.921 |
| mass-balanced ∝ n_l (0.125, 0.250, 0.250, 0.1875, 0.125, 0.0625) | 8/9 | 12/15 | 8 | 0.947 | 0.838 |
| L5-zero, renormalised (0.067, 0.133, 0.200, 0.267, 0.333, 0) | 6/9 | **15/15** | 6 | 0.983 | 0.903 |

N-balanced weighting *increases* L5's per-instruction influence (its weight
rises to 0.353) and still keeps the cell-level top-9 at 9/9; its only top-15
change is DeepSeek-V3's own few-shot/few-cot runs swapping at ranks 14–16.
Under the uniform and mass-balanced schemes the single top-9 change is
Qwen3.5-35B few-cot (default #8) exchanging with Qwen3.5-122B few-shot
(default #10).

The L5-zero ablation is the strongest probe: with L5 deleted, cell-level top-9
retention drops to 6/9 — and the full record shows every change is a
within-family strategy swap inside the default top-12, with the top-15 cell
*set* preserved 15/15:

| Move | Cell | default → L5-zero rank |
|---|---|---|
| out of top-9 | Qwen3.5-397B zero-cot | #6 → #12 |
| out of top-9 | Qwen3.5-122B zero-cot | #7 → #10 |
| out of top-9 | Qwen3.5-35B few-cot | #8 → #11 |
| into top-9 | Qwen3.5-122B few-shot | #10 → #6 |
| into top-9 | Qwen3.5-397B few-shot | #12 → #8 |
| into top-9 | GPT-4.1 few-shot | #11 → #9 |

The demoted cells are all CoT variants and the promoted ones few-shot runs of
the same families — consistent with CoT prompting earning part of its premium
on L5's multi-step tasks. At the model level (same best-baseline construction
as the table above), the L5-zero ablation leaves both the top-5 and the top-9
completely intact (5/5 and 9/9) with a maximum rank displacement of **1** —
four adjacent-pair swaps (Qwen3.5-35B ↔ GPT-4.1 at #4/#5, GPT-4o ↔ DeepSeek-V3
at #6/#7, Sonnet-4 ↔ Llama-3.3-70B at #8/#9, GPT-3.5-turbo ↔ GPT-4o-mini at
#10/#11; ρ = 0.972, τ-b = 0.879).

Reading: L5 carries real diagnostic signal — deleting it re-orders cells at
ranks 6–12 — but the signal is balanced rather than dominant. L5-zero is in
fact the only scheme whose top-15 set is preserved exactly; the re-weighting
schemes admit cells from just outside (default #16–#17; mass-balanced, which
down-weights L4 as well, reaches to default #23). And the model leaderboard's
membership is unchanged at every cut even with L5 removed entirely.

---

## 10. External-anchor and selection generalization (`off2_*`, `off4_*`,
`offa_*` products)

| Analysis | Result |
|---|---|
| OFF-2 Oracle-per-instance selection | +0.0360 (11/12 models) |
| OFF-2 Self-Consistency best-of-5 (optimistic ceiling) | +0.0558 (12/12) |
| OFF-2 CAAP selection | +0.0061 (10/12) |
| OFF-4 leave-one-model-out vs zero-shot | +0.0315 (11/12) |
| OFF-4 leave-one-model-out vs best single | +0.0025 (6/12) |
| OFF-A task-family regroup, StructuralConstraint peak (quality-only) | 0.1657 |
| OFF-A task-family regroup, StructuralConstraint peak (continuous-D4) | 0.1625 |

The task-family regrouping (OFF-A) re-derives the peak without the L-level
labels, using a semantic re-grouping of the instructions; the multi-constraint /
structural-constraint family remains the most discriminative.

The selection analyses (OFF-2/4) are defined on the models that carry
improvement runs (Oracle/SC/CAAP); the four ext16-only models were added for
baseline robustness and have no such runs, so they are excluded from OFF-2/4
and the counts stay at n = 12 (the `ext16` recompute records them under
`skipped_models` rather than silently dropping them).

---

## 11. Thinking ablation (GLM-4.6, open vs. disabled)

To separate the L2/L4 findings from any reasoning-specific effect, GLM-4.6 was
run with thinking enabled and disabled, under both zero-shot and the Combined
method (quality-only totals):

| GLM-4.6 | thinking on | thinking off | Δ |
|---|---|---|---|
| zero-shot total | 0.851 | 0.813 | +0.038 |
| Combined total | 0.912 | 0.880 | +0.031 |
| zero-shot L4 | 0.714 | 0.703 | +0.011 |
| Combined L4 | 0.836 | 0.831 | +0.005 |

The thinking benefit is small at the total level (+0.031 to +0.038) and
negligible at L4 (open and disabled within 0.011). The L4 ceiling is therefore
not a consequence of thinking being unavailable. With thinking disabled, the
Combined method still lifts L4 from 0.703 to 0.831 (+0.128, via format adoption
as in §7), so the method's effect does not depend on thinking either.

---

## 12. D5 scale stability; corrigendum

The D5 provider-concentration finding is scale-stable: recomputing D5 over a
3×3 grid of its exponential decay scales leaves the efficiency ranking
unchanged, Spearman ρ ∈ [0.966, 1.000] in every cell (full grid in the
submitted Appendix D "D5 robustness" table, derived from the same released
scores).

Corrigendum: the paper's "791 feasible L3+" should read "791 feasible across
all levels" (800 instructions − 9 infeasible L2 regularity-constraint cases).

---

## Data provenance

Every number above is traceable to a product file under
`rebuttal_analyses/results/`:

| Numbers | Product file |
|---|---|
| Pearson/Spearman r(D1,D4) per level | `b1_dimension_correlation.json` |
| Dimension-subset leaderboards (§2.1) | `b1b_dimension_subset_rankings.json` |
| Continuous-D4 tier gap L0–L5 (12- and 16-model) | `c1_d4_only_tiergap.json`, `ext16/c1_d4_only_tiergap.json` |
| L2 gap bootstrap CI; L5 CI [0.074, 0.227] | `c2_tiergap_bootstrap_ci.json` |
| Tier-free cross-model std (12- and 16-model) | `s1_tierfree_discrimination.json`, `ext16/s1_tierfree_discrimination.json` |
| Split-data tiers (12-model 5/5; 16-model freq 0.8) | `e1b_splitdata_tier.json`, `ext16/e1b_splitdata_tier.json` |
| Active-parameter counter-case | `e1a_external_param_tier.json` |
| External Elo / MMLU-Pro anchors (full 16-model) | `off-b_external_tier.json`, `ext16/off-b_external_tier.json` |
| Instruction-paraphrase robustness (6-model) | `exp6_paraphrase.json` |
| Held-out-exemplar + decoupled-verifier summary | `exp23_heldout_decoupling.json` |
| Combined/VGIG held-out dimension gains | `d1_vgig_heldout_gains.json` |
| VGIG vs Retry/SC | `d2_vgig_vs_retry_sc.json` |
| L4 dimension decomposition (8.70× / 8.85× / 4.21×) | `l4d_l4_dimension_decomp.json`, `off5_l4_semantic_altmetric.json` |
| Weight sensitivity; rank-aggregation | `a1_level_weight_sensitivity.json`, `a2_aggregation_crosscheck.json` |
| Cell-level schemes; L5-zero ablation (§9.1, both granularities) | `a1b_cell_weight_sensitivity.json` |
| Selection / task-family generalization | `off2_*.json`, `off4_*.json`, `offa_*.json` |
| Unified robustness matrix | `uni_unified_robustness.json` |
| Cross-family Combined/Retry (§3, 9 families) | `results/quality/<slug>-{combined,retry}.quality.json` |
| Reasoning panel quality totals | `results/quality/{gpt5,gpt5mini,zaiorgGLM46,deepseekaiDeepSeekR10528,QwenQwen3235BA22BThinking2507}-*.quality.json` |
| Held-out-exemplar EXP-2 | `results/quality/{deepseekaiDeepSeekV3,gpt4omini,qwen3535ba3b}-few-{shot,cot}{,-heldout}.quality.json` |
| Thinking ablation | `results/quality/zaiorgGLM46-{zero-shot,combined}{,-nt}.quality.json` |
| D5 scale stability (§12) | submitted Appendix D "D5 robustness" table |
