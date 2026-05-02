# Installation

### System requirements

- **Python**: 3.10 or 3.11 (3.12 also works; tested on 3.10.12)
- **OS**: Linux, macOS, or Windows (tested on Ubuntu 22.04 and Windows 11)
- **RAM**: 8 GB minimum (16 GB recommended for D3 embedding metric on large graphs)
- **GPU** (optional): only required for the D3 embedding metric (lightweight 2-layer GCN); CPU fallback available; tested on a single laptop NVIDIA RTX 4070 Laptop (8 GB VRAM)
- **API keys** (optional): only needed if you want to run new generations against commercial LLMs. Re-evaluating cached outputs needs **no API key**.

### Core install (5 lines)

```bash
# 0. Windows users: set these once per shell to avoid two well-known issues
#    (a) OMP runtime conflict between numpy/torch causing "Initializing
#        libiomp5md.dll, but found libiomp5md.dll already initialized" abort;
#    (b) Python's default GBK encoding on Windows crashing on UTF-8 JSON.
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONIOENCODING=utf-8

# 1. Get the code (this zip; or clone the repo)
cd graphinstruct-supplementary

# 2. Create a virtual environment (recommended)
python -m venv venv && source venv/bin/activate     # Linux/macOS
# or:  python -m venv venv && venv\Scripts\activate # Windows

# 3. Install with core dependencies
pip install -e .

# 4. Verify install: 418 unit tests should all pass (~30 s)
python -m unittest discover -v -s tests

# 5. (optional) Install full dependencies for D2/D3 metrics
pip install -e ".[full]"
```

### Optional dependencies

The default install includes only the lightweight core (`networkx`, `numpy`, `tiktoken`, `matplotlib`). Heavy dependencies are gated behind the `[full]` extras:

| Extra | Adds | Needed for |
|-------|------|-----------|
| `[full]` | `transformers`, `torch`, `torch_geometric`, `igraph`, `plotly` | D2 G-BERTScore (BERT model), D3 Node-Classification Gap (GCN), interactive Pareto plot |

If you only want to run **D1, D4, D5** evaluation (the dimensions used at L0–L2 and the dimensions reported in main paper Tables 1–3), the core install suffices and no GPU is needed.

### Calling commercial LLMs

To run new (model, strategy) generations, set the relevant API key as an environment variable. We support OpenAI, Anthropic, Alibaba Bailian, and DeepSeek out of the box:

```bash
export OPENAI_API_KEY=sk-...               # OpenAI: GPT-3.5/4o-mini/4o/4.1
export ANTHROPIC_API_KEY=sk-ant-...        # Anthropic: Sonnet-4 / Sonnet-4.6
export DASHSCOPE_API_KEY=sk-...            # Alibaba Bailian: Qwen3.5-{35B,122B,397B}
export DEEPSEEK_API_KEY=sk-...             # DeepSeek: V3
```

**Cost transparency.** Reproducing the full 12-LLM × 4-strategy survey (180 K outputs) would cost roughly USD ~600 in API spend at 2026 H1 prices, dominated by Anthropic and OpenAI per the per-provider breakdown reported in paper Appendix H. Reproducing a single (model, strategy) cell on GPT-4o-mini costs ~USD 4 and takes ~1 hour.

### Troubleshooting

- **`ImportError: tiktoken`**: install with `pip install tiktoken`.
- **OMP errors on Windows**: prepend `KMP_DUPLICATE_LIB_OK=TRUE` to your environment.
- **D3 metrics fail with CUDA OOM**: pass `--batch-size 16` (default 32); the laptop tests fit at batch 32 on 8 GB.
