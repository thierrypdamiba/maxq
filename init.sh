#!/bin/bash
# MaxQ Environment Setup Script
# This script sets up the development environment for MaxQ
# Run: ./init.sh

set -e  # Exit on error

echo "========================================"
echo "MaxQ Environment Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}[1/7] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python 3.10+ required. Found: $PYTHON_VERSION${NC}"
    echo "Install Python 3.10+ and try again."
    exit 1
fi
echo -e "${GREEN}Python $PYTHON_VERSION detected${NC}"

# Create virtual environment if it doesn't exist
echo -e "\n${YELLOW}[2/7] Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Created new virtual environment${NC}"
else
    echo -e "${GREEN}Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate
echo -e "${GREEN}Activated virtual environment${NC}"

# Install dependencies
echo -e "\n${YELLOW}[3/7] Installing dependencies...${NC}"
if command -v uv &> /dev/null; then
    uv pip install -e ".[dev]"
else
    pip install -e ".[dev]"
fi
echo -e "${GREEN}Dependencies installed${NC}"

# Create .env file if it doesn't exist
echo -e "\n${YELLOW}[4/7] Setting up environment variables...${NC}"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}Created .env from .env.example${NC}"
        echo -e "${YELLOW}Please edit .env with your credentials:${NC}"
        echo "  - QDRANT_URL: Your Qdrant Cloud URL"
        echo "  - QDRANT_API_KEY: Your Qdrant Cloud API key"
    else
        cat > .env << 'EOF'
# MaxQ Environment Configuration
# Get your Qdrant Cloud credentials at https://cloud.qdrant.io

# Qdrant Cloud (required)
QDRANT_URL=
QDRANT_API_KEY=

# Optional: OpenAI for LLM features (HyDE, RAG)
OPENAI_API_KEY=

# Optional: Logging level
LOG_LEVEL=INFO
EOF
        echo -e "${YELLOW}Created .env template${NC}"
        echo -e "${YELLOW}Please edit .env with your credentials${NC}"
    fi
else
    echo -e "${GREEN}.env file already exists${NC}"
fi

# Create runs directory
echo -e "\n${YELLOW}[5/7] Creating runs directory...${NC}"
mkdir -p runs
echo -e "${GREEN}Runs directory ready${NC}"

# Create ~/.maxq directory for global config
echo -e "\n${YELLOW}[6/7] Setting up MaxQ app directory...${NC}"
mkdir -p ~/.maxq
echo -e "${GREEN}~/.maxq directory ready${NC}"

# Initialize git repository if not already initialized
echo -e "\n${YELLOW}[7/7] Setting up git repository...${NC}"
if [ ! -d ".git" ]; then
    git init
    echo -e "${GREEN}Initialized git repository${NC}"
else
    echo -e "${GREEN}Git repository already exists${NC}"
fi

# Run database migrations
echo -e "\n${YELLOW}Running database migrations...${NC}"
python -c "from maxq.db.migrations import run_migrations; run_migrations()"
echo -e "${GREEN}Database migrations complete${NC}"

# Verify installation
echo -e "\n${YELLOW}Verifying installation...${NC}"
if python -c "import maxq; print(f'MaxQ version: {maxq.__version__ if hasattr(maxq, \"__version__\") else \"0.0.1\"}')" 2>/dev/null; then
    echo -e "${GREEN}MaxQ installed successfully${NC}"
else
    echo -e "${RED}Warning: Could not verify MaxQ installation${NC}"
fi

# Print summary
echo -e "\n========================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your Qdrant credentials"
echo "  2. Activate the environment: source venv/bin/activate"
echo "  3. Run the doctor check: maxq doctor"
echo "  4. Start the worker: maxq worker start"
echo "  5. Start the API server: uvicorn maxq.server.main:app --port 8000"
echo ""
echo "For more information, see README.md"
