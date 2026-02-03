#!/bin/bash
# Setup script for Test-time Recursive Thinking (TRT)
# This script checks prerequisites and installs dependencies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "TRT Environment Setup"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python >= 3.10 is required. Found Python $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}Python $PYTHON_VERSION detected${NC}"

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$CUDA_VERSION" ]; then
        echo -e "${GREEN}CUDA available (Driver: $CUDA_VERSION)${NC}"
    else
        echo -e "${YELLOW}Warning: nvidia-smi found but no GPU detected${NC}"
    fi
else
    echo -e "${YELLOW}Warning: CUDA not detected. GPU acceleration will not be available.${NC}"
    echo -e "${YELLOW}CUDA is recommended for AIME experiments with vLLM.${NC}"
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create virtual environment if requested
if [ "$1" = "--venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    VENV_DIR="${SCRIPT_DIR}/venv"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo -e "${GREEN}Virtual environment created at $VENV_DIR${NC}"
    echo "To activate: source $VENV_DIR/bin/activate"
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install PyTorch (if CUDA available, install CUDA version)
echo ""
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    pip install torch -q
else
    pip install torch --index-url https://download.pytorch.org/whl/cpu -q
fi
echo -e "${GREEN}PyTorch installed${NC}"

# Install AIME dependencies
echo ""
echo "Installing AIME dependencies..."
if [ -f "${SCRIPT_DIR}/AIME/requirements.txt" ]; then
    pip install -r "${SCRIPT_DIR}/AIME/requirements.txt" -q
    echo -e "${GREEN}AIME dependencies installed${NC}"
else
    echo -e "${YELLOW}Warning: AIME/requirements.txt not found${NC}"
fi

# Install LiveCodeBench
echo ""
echo "Installing LiveCodeBench..."
if [ -f "${SCRIPT_DIR}/LiveCodeBench/pyproject.toml" ]; then
    cd "${SCRIPT_DIR}/LiveCodeBench"
    pip install -e . -q
    cd "${SCRIPT_DIR}"
    echo -e "${GREEN}LiveCodeBench installed${NC}"
else
    echo -e "${YELLOW}Warning: LiveCodeBench/pyproject.toml not found${NC}"
fi

# Install MCP server dependencies
echo ""
echo "Installing MCP server dependencies..."
pip install mcp fastmcp aiofiles orjson -q
echo -e "${GREEN}MCP dependencies installed${NC}"

# Summary
echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Set environment variables (for LiveCodeBench):"
echo "   export OPENAI_API_KEY=\"your-azure-openai-api-key\""
echo "   export AZURE_OPENAI_ENDPOINT=\"your-azure-openai-endpoint\""
echo ""
echo "2. For AIME experiments:"
echo "   cd AIME/bash_scripts"
echo "   bash qwen3.sh"
echo ""
echo "3. For LiveCodeBench experiments:"
echo "   # Terminal 1: Start MCP server"
echo "   cd LiveCodeBench/kflow_mcp && bash start_server.sh"
echo ""
echo "   # Terminal 2: Run evaluation"
echo "   cd LiveCodeBench/bash_scripts && bash kflow_o4-mini.sh"
echo ""
