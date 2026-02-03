#!/bin/bash
# Start the TRT Knowledge Flow MCP server

export KNOWLEDGE_BASE_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export KFLOW_PORT=${1:-${KFLOW_PORT:-8000}}

python knowledge_flow_server.py