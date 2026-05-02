# GraphInstruct — Datasheet for Datasets / 数据集说明书


> This datasheet follows the template of Gebru et al. (2021), *"Datasheets for Datasets"*. The full version with extensive answers under each of the 7 sections is in **paper Appendix I**. This file in the supplementary repeats the most reviewer-relevant questions for quick reference.

### 1. Motivation

**For what purpose was the dataset created?**
GraphInstruct was created to provide the first **progressive-complexity diagnostic benchmark** for LLM graph generation. Existing graph-LLM benchmarks stratify along graph-type, task-domain, or classical-algorithm axes, all of which average over the structural-complexity dimension that actually governs failure. GraphInstruct closes this diagnostic gap by stratifying outputs into six progressively-constrained complexity levels and scoring along five evaluation dimensions targeting structurally distinct failure modes.

**Who created the dataset and on behalf of which entity?**
*Anonymized for double-blind review.* The benchmark was created by university researchers in machine learning, with no commercial sponsor.

**Who funded the creation?**
*Anonymized for double-blind review.*

### 2. Composition

**What do the instances represent?**
Each instance is a (instruction, reference graph(s), constraint specification) tuple. The instruction is a natural-language query asking an LLM to generate a graph with specified properties; the reference graph(s) are algorithmically synthesized constraint-satisfying graphs; the constraint specification enumerates explicit and implicit structural constraints.

**How many instances?**
800 instructions, 1,582 reference solutions, 4,163 distributional reference graphs (3,115 L3 synthetic + 1,048 L4 real).

**Does the dataset contain all possible instances or a sample?**
A stratified sample, with explicit per-(level × size × constraint-type) cell coverage of ≥15 instances. We deliberately do not aim for full combinatorial constraint coverage; the 800-instance budget is calibrated to be reproducibly evaluatable on a single laptop in ~1 hour and to fit within commercial-API budget norms.

**Does the data contain confidential information?**
No. All instructions are hand-authored; all reference graphs are either algorithmically synthesized or drawn from public datasets (see `DATA_LICENSE.md`).

**Are there errors / sources of noise?**
We document 9 deliberately infeasible L2 instructions (`feasible=false`) for which no constraint-satisfying graph exists. Beyond these, the benchmark passed six review rounds (three consecutive clean passes) and a 418-unit-test parser / validator suite.

### 3. Collection process

**How was the data collected?**
Three-stage pipeline:
- **Stage 1 (Template authoring):** 40 hand-designed templates cover L0–L5 with parameter slots; two authors reviewed each for linguistic clarity and constraint well-formedness.
- **Stage 2 (Parameter sampling):** stratified sampling yields 800 instructions with balanced coverage of graph types, sizes, and constraint counts.
- **Stage 3 (Reference synthesis):** 2 reference graphs synthesized per feasible instruction by constraint-satisfying algorithms (NetworkX `random_labeled_tree`, k-core peeling, calibrated attribute sampling, etc.).

**Time frame.** Data construction occurred between 2025 December and 2026 March.

### 4. Preprocessing / cleaning

All reference graphs pass round-trip parse → serialize → re-parse; size-normalized by BFS/random walk to fit instruction size buckets; deduplicated by Weisfeiler-Lehman graph hash. L4 real-world subsets are stripped of attributes other than what D2/D3 need.

### 5. Uses

**Has the dataset been used for any other tasks?** No prior use (this is the first release).

**What other tasks could the dataset be used for?**
- Diagnostic benchmark for LLM graph-generation capability (intended use)
- Evaluation harness for prompt-engineering, retrieval-augmented, and verification-guided generation methods
- Stratified data for training graph-aware code generation models

**Are there tasks the dataset should NOT be used for?**
- The synthetic graphs are not photorealistic data and are not suitable for training graph foundation models intended for real-world deployment.
- L4 real-world subsets carry their upstream license restrictions; do not redistribute beyond the upstream terms.

### 6. Distribution

**How is the dataset distributed?**
Public release on GitHub (link in main paper). Croissant metadata (JSON-LD) provided. Code released under MIT, instructions / synthetic graphs under CC-BY-4.0, L4 real-world subsets retain upstream licenses.

**Will the dataset be public?** Yes, post-acceptance.

### 7. Maintenance

**Who maintains the dataset?**
The authors (anonymized; identities to be filled in upon acceptance).

**How will updates be communicated?**
Versioned releases on GitHub with semantic-versioned tags. Major changes will trigger a new DOI on Zenodo.

**Will erroneous / outdated data be corrected?**
Yes; corrections will be issued as patch releases.
