#!/bin/bash
# MaxQ Development Setup Script
# Usage: ./tools/dev.sh

set -e

echo "=== MaxQ Development Setup ==="
echo

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION+ required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
echo "✓ Virtual environment"

# Activate venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
if command -v uv &> /dev/null; then
    uv pip install -e ".[dev]"
else
    pip install -e ".[dev]"
fi
echo "✓ Dependencies installed"

# Check for .env
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ Created .env from .env.example"
        echo "  → Edit .env with your Qdrant credentials"
    else
        echo "⚠ No .env file found"
    fi
else
    echo "✓ .env exists"
fi

# Check Qdrant credentials
if [ -z "$QDRANT_URL" ] && ! grep -q "QDRANT_URL=" .env 2>/dev/null; then
    echo "⚠ QDRANT_URL not set"
    echo "  → Set in .env or environment"
fi

# Install frontend dependencies (optional)
if [ -d "studio-ui" ]; then
    echo
    read -p "Install frontend dependencies? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd studio-ui
        if command -v pnpm &> /dev/null; then
            pnpm install
        else
            npm install
        fi
        cd ..
        echo "✓ Frontend dependencies installed"
    fi
fi

echo
echo "=== Setup Complete ==="
echo
echo "Activate environment:"
echo "  source venv/bin/activate"
echo
echo "Run commands:"
echo "  maxq doctor      # Health check"
echo "  maxq demo        # Load sample data"
echo "  maxq studio      # Start web UI"
echo
echo "Start API:"
echo "  uvicorn maxq.server.main:app --port 8000"
echo
