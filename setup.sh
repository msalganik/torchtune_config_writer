#!/bin/bash
set -e

echo "🚀 Setting up torchtune_config_writer..."
echo

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo
fi

# Create virtual environment
echo "🔨 Creating virtual environment..."
uv venv
echo

# Activate and install
echo "📥 Installing dependencies..."
source .venv/bin/activate
uv pip install -e ".[dev]"
echo

# Create scratch directory for experiments
mkdir -p scratch

echo "🧪 Verifying installation..."
echo

# Check if torchtune is working
if tune ls > /dev/null 2>&1; then
    echo "  ✓ torchtune installed and working"
else
    echo "  ❌ torchtune not working. Check installation."
    exit 1
fi

# Check if pytest can collect tests
if pytest --collect-only > /dev/null 2>&1; then
    echo "  ✓ pytest installed and working"
else
    echo "  ⚠️  pytest not working. Tests may not run."
fi

echo
echo "✅ Setup complete!"
echo
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
