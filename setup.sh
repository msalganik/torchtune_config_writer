#!/bin/bash
set -e

echo "🚀 Setting up torchtune_config_writer..."
echo

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo

    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"

    # Verify uv is now available
    if ! command -v uv &> /dev/null; then
        echo "❌ uv installation failed or not in PATH"
        echo "Please add $HOME/.local/bin to your PATH and re-run this script"
        exit 1
    fi
    echo "✓ uv installed successfully"
fi

# Create package directory structure
echo "📁 Creating package structure..."
mkdir -p torchtune_config_writer tests/fixtures scratch

# Create __init__.py if it doesn't exist
if [ ! -f torchtune_config_writer/__init__.py ]; then
    cat > torchtune_config_writer/__init__.py << 'EOF'
"""Torchtune Config Writer - Tool for generating torchtune YAML configs."""

__version__ = "0.1.0"
EOF
    echo "✓ Created package __init__.py"
fi
echo

# Create virtual environment
echo "🔨 Creating virtual environment..."
uv venv
echo

# Activate and install
echo "📥 Installing dependencies..."
source .venv/bin/activate
uv pip install -e ".[dev]"
echo

echo "🧪 Verifying installation..."
echo

# Check if torchtune is working
if tune --help > /dev/null 2>&1; then
    echo "  ✓ torchtune CLI installed and working"

    # Count available recipes to verify full functionality
    RECIPE_COUNT=$(tune ls 2>/dev/null | wc -l)
    if [ "$RECIPE_COUNT" -gt 10 ]; then
        echo "  ✓ torchtune recipes available ($RECIPE_COUNT lines listed)"
    fi
else
    echo "  ❌ torchtune not working. Check installation."
    echo "  Hint: Try activating the venv and running 'uv pip install torchao'"
    exit 1
fi

# Check if pytest can collect tests
if pytest --collect-only > /dev/null 2>&1; then
    echo "  ✓ pytest installed and working"
else
    echo "  ⚠️  pytest not working. Tests may not run."
fi

# Check Python can import the package
if python -c "import torchtune_config_writer" 2>/dev/null; then
    echo "  ✓ torchtune_config_writer package importable"
else
    echo "  ⚠️  Package import failed (this is OK if you haven't implemented code yet)"
fi

echo
echo "✅ Setup complete!"
echo
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo
echo "Next steps:"
echo "  - Run 'tune ls' to see available configs"
echo "  - Run 'tune cp llama3_1/8B_lora_single_device scratch/sample.yaml' to see a sample config"
echo "  - Run 'pytest' to run tests (when created)"
