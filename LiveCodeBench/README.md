# LiveCodeBench Evaluation

Code generation evaluation framework with Test-time Recursive Thinking (TRT) agentic approach.

## What is TRT?

**Test-time Recursive Thinking (TRT)** enables LLMs to self-improve during inference through iterative reflection. The model generates solution candidates, selects the best ones, and reflects on successes and failures to extract knowledge that improves subsequent attempts—all without external feedback.

## Prerequisites

- Python >= 3.10
- Azure OpenAI API access
- For MCP server: `pip install mcp fastmcp aiofiles orjson`

## Installation

```bash
cd LiveCodeBench
pip install -e .
```

## Environment Variables

Set the following environment variables before running experiments:

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | Yes | Azure OpenAI API key | - |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI endpoint URL | - |
| `MCP_SERVER_URL` | For TRT | TRT MCP server URL | `http://localhost:8000/sse` |
| `STREAM_DEBUG` | No | Enable debug logging | `false` |

Example:
```bash
export OPENAI_API_KEY="your-azure-openai-api-key"
export AZURE_OPENAI_ENDPOINT="your-azure-openai-endpoint-url"
```

## Running Experiments

### Baseline Experiments

Run standard LLM evaluation without TRT:

```bash
cd bash_scripts
bash baseline_o4_mini.sh   # o4-mini model
bash baseline_o3_high.sh   # o3 model
```

### TRT Experiments

Test-time Recursive Thinking (TRT) uses an agentic approach with an MCP server for iterative problem-solving.

**Step 1: Start the TRT Server**
```bash
cd kflow_mcp
bash start_server.sh
```

**Step 2: Run the TRT Experiment** (in a separate terminal)
```bash
cd bash_scripts
bash kflow_o4-mini.sh   # o4-mini with TRT
bash kflow_o3.sh        # o3 with TRT
```

## Project Structure

```
LiveCodeBench/
├── bash_scripts/           # Experiment launch scripts
│   ├── baseline_o4_mini.sh
│   ├── baseline_o3_high.sh
│   ├── kflow_o4-mini.sh
│   └── kflow_o3.sh
├── kflow_mcp/              # TRT MCP server
│   ├── knowledge_flow_server.py
│   └── start_server.sh
├── lcb_runner/             # Main evaluation runner
├── assets/                 # Images and figures
└── pyproject.toml
```

## Configuration Options

Key command-line arguments for the runner:

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name/identifier | Required |
| `--scenario` | Evaluation scenario | `codegeneration` |
| `--max_tokens` | Maximum token limit | `200000` |
| `--n` | Number of samples | `1` |
| `--trt_rounds` | Number of TRT iterations (agentic only) | `8` |
| `--roll_out_n` | Number of rollouts per problem | `2` |
| `--difficulty` | Problem difficulty filter | `hard` |
| `--multiprocess_oai` | Number of parallel API calls | `16` |
| `--openai_timeout` | API timeout in seconds | `1800` |

## Troubleshooting

### MCP Server Connection Issues

**Symptom**: "Connection refused" or timeout errors when running TRT experiments.

**Solutions**:
1. Ensure the MCP server is running: `cd kflow_mcp && bash start_server.sh`
2. Verify `MCP_SERVER_URL` is set correctly: `echo $MCP_SERVER_URL`
3. Check if port 8000 is available: `lsof -i :8000`
4. Check server logs for errors

### API Timeout Errors

**Symptom**: Requests timing out with Azure OpenAI API.

**Solutions**:
1. Increase timeout: Add `--openai_timeout 3600` to your command
2. Reduce parallel requests: Use `--multiprocess_oai 8` instead of 16
3. Check your Azure OpenAI quota and rate limits

### Missing Dependencies

**Symptom**: Import errors when running experiments.

**Solutions**:
```bash
# Reinstall the package
pip install -e .

# Install MCP dependencies
pip install mcp fastmcp aiofiles orjson
```
