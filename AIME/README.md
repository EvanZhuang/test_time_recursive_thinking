# AIME Mathematical Reasoning Evaluation

Evaluation framework for LLMs on AIME (American Invitational Mathematics Examination) problems.

## Installation

```bash
cd AIME
pip install -r requirements.txt
```

### Requirements

- vllm==0.10.2
- transformers==4.57.0
- nltk>=3.9.2
- openai==2.1.0
- openai-harmony==0.0.4

## Running Experiments

```bash
cd bash_scripts
bash qwen3.sh     # Qwen3 model evaluation
bash gpt_oss.sh   # GPT model evaluation
```

## Configuration

Edit the bash scripts to adjust model and generation parameters:

| Parameter | Description |
|-----------|-------------|
| `--model_name` | Model identifier (e.g., `Qwen/Qwen3-235B-A22B-Thinking-2507`) |
| `--max_new_tokens` | Maximum tokens to generate |
| `--temperature` | Sampling temperature |
| `--reflex_size` | Reflection size for TRT |

## Project Structure

```
AIME/
├── bash_scripts/           # Experiment launch scripts
│   ├── qwen3.sh
│   └── gpt_oss.sh
├── scripts/                # Python evaluation scripts
├── requirements.txt        # Python dependencies
└── requirements_full.txt   # Extended dependencies
```

## Environment Variables

For Hugging Face model downloads:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
```

For vLLM optimization:
```bash
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false
```
