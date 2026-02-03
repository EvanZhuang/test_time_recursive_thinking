<h1 align="center">Test-time Recursive Thinking (TRT)</h1>
<p align="center"><b>Self-Improvement without External Feedback</b>
(<a href="https://arxiv.org/abs/">arXiv</a>)</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg">
  <img src="https://img.shields.io/badge/python-3.10+-blue">
</p>

<!-- Add TRT illustration: <img src="assets/trt.png" alt="TRT Illustration" width="600"> -->

---

## What is TRT?

**Test-time Recursive Thinking (TRT)** is an agentic framework that enables LLMs to self-improve during inference through iterative reflection—without requiring external feedback or reward signals.

TRT operates in three stages:

1. **Generate**: The model produces multiple solution candidates for a given problem
2. **Select**: Solutions are evaluated and the best candidates are identified using self-consistency or verification
3. **Reflect**: The model analyzes successful and failed attempts to extract generalizable insights, which inform subsequent generation rounds

This recursive process allows the model to accumulate knowledge within a session, progressively improving solution quality through self-directed learning.

---

## Installation

### Prerequisites

- Python >= 3.10
- CUDA-compatible GPU (recommended for vLLM-based experiments)
- Azure OpenAI API access (for LiveCodeBench experiments)

### Quick Install

```bash
git clone https://github.com/YufanZhuang/test-time-recursive-thinking.git
cd test-time-recursive-thinking
./setup_env.sh
```

### Manual Install

```bash
# Install AIME dependencies
cd AIME
pip install -r requirements.txt

# Install LiveCodeBench
cd ../LiveCodeBench
pip install -e .

# Install MCP server dependencies (for TRT agentic mode)
pip install mcp fastmcp aiofiles orjson
```

---

## Quick Start

### AIME Mathematical Reasoning

```bash
cd AIME/bash_scripts
bash qwen3.sh     # Run Qwen3-235B evaluation
bash gpt_oss.sh   # Run GPT model evaluation
```

### LiveCodeBench Code Generation

**Step 1: Set environment variables**
```bash
export OPENAI_API_KEY="your-azure-openai-api-key"
export AZURE_OPENAI_ENDPOINT="your-azure-openai-endpoint"
```

**Step 2: Start the TRT MCP server**
```bash
cd LiveCodeBench/kflow_mcp
bash start_server.sh
```

**Step 3: Run TRT evaluation** (in a separate terminal)
```bash
cd LiveCodeBench/bash_scripts
bash kflow_o4-mini.sh   # o4-mini with TRT
bash kflow_o3.sh        # o3 with TRT
```

---

## Configuration Options

### AIME Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | Model identifier (e.g., `Qwen/Qwen3-235B-A22B-Thinking-2507`) | Required |
| `--max_new_tokens` | Maximum tokens to generate | `262144` |
| `--temperature` | Sampling temperature | `0.6` |
| `--reflex_size` | Number of reflection samples (Maj@N) | `64` |

### LiveCodeBench Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model name/identifier | Required |
| `--scenario` | Evaluation scenario | `codegeneration` |
| `--max_tokens` | Maximum token limit | `200000` |
| `--trt_rounds` | Number of TRT iterations | `8` |
| `--roll_out_n` | Number of rollouts per problem | `2` |
| `--difficulty` | Problem difficulty filter | `hard` |

---

## Evaluations

### AIME Mathematical Reasoning

TRT achieves **100% accuracy** on both AIME-24 and AIME-25 benchmarks:

### LiveCodeBench Code Generation (Hard Problems)

TRT provides significant improvements on hard coding problems:

| Model | Baseline | TRT | Improvement |
|-------|----------|-----|-------------|
| o4-mini (high) | 63.5% | 73.9% | **+10.4pp** |
| o3 (high) | 57.1% | 71.9% | **+14.8pp** |

---

## Project Structure

```
test-time-recursive-thinking/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── setup_env.sh              # Environment setup script
├── assets/                   # Images and figures
├── AIME/                     # AIME mathematical reasoning
│   ├── bash_scripts/         # Experiment launch scripts
│   ├── scripts/              # Python evaluation scripts
│   ├── requirements.txt      # Dependencies
│   └── README.md
└── LiveCodeBench/            # Code generation evaluation
    ├── bash_scripts/         # Experiment launch scripts
    ├── kflow_mcp/            # TRT MCP server
    ├── lcb_runner/           # Main evaluation runner
    ├── pyproject.toml        # Package configuration
    └── README.md
```

---

## Questions?

If you have any questions related to the code or the paper, feel free to reach out to us at y5zhuang@ucsd.edu.

---

## Citation

If you find our paper and code useful, please cite us:

```bibtex
@article{zhuang2025trt,
  title={Test-time Recursive Thinking: Self-Improvement without External Feedback},
  author={Zhuang, Yufan and Liu, Liyuan and Singh, Chandan and Shang, Jingbo and Gao, Jianfeng},
  journal={arXiv preprint arXiv:},
  year={2025}
}
```
