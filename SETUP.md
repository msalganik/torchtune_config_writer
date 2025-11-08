# Development Environment Setup

## Quick Start (Returning Users)

```bash
source .venv/bin/activate  # Activate environment
```

## Prerequisites

- Python 3.10 or later
- uv package manager (script will install if missing)
- Git (for cloning torchtune if needed)

## First-Time Setup

### Option 1: Automated Setup (Recommended)

```bash
./setup.sh
source .venv/bin/activate
```

**What the script does:**
- Installs uv package manager if not present
- Adds uv to PATH automatically
- Creates package directory structure
- Creates virtual environment
- Installs all dependencies including torchao
- Verifies installation

### Option 2: Manual Setup

```bash
# 1. Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Add uv to PATH (required after first install)
export PATH="$HOME/.local/bin:$PATH"

# 3. Create package directory structure
mkdir -p torchtune_config_writer tests/fixtures scratch

# 4. Create package __init__.py file
cat > torchtune_config_writer/__init__.py << 'EOF'
"""Torchtune Config Writer - Tool for generating torchtune YAML configs."""

__version__ = "0.1.0"
EOF

# 5. Create virtual environment
uv venv

# 6. Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows

# 7. Install in editable mode with development dependencies
uv pip install -e ".[dev]"
```

**What does this do?**
- `-e` installs in "editable" mode - changes to code are immediately available
- `.[dev]` installs core dependencies + development tools (pytest, coverage)
- Creates necessary directory structure before installation

## Verify Installation

```bash
# Check torchtune is installed
tune --help

# List available recipes
tune ls

# Run tests to verify everything works
pytest
```

## Project Structure After Setup

```
torchtune_config_writer/
├── .venv/                       # Virtual environment (gitignored)
├── .gitignore                   # Git ignore rules
├── CLAUDE.md                    # Guiding principles
├── SPEC.md                      # Technical specification
├── SETUP.md                     # This file
├── pyproject.toml               # Project configuration and dependencies
├── config_builder.py            # Main implementation (to be created)
└── tests/                       # Test suite
```

## Next Steps

After setup is complete:
1. Explore available torchtune configs: `tune ls`
2. Examine a sample config: `tune cp llama3_1/8B_lora_single_device scratch/sample.yaml`
3. Run tests: `pytest`
4. Start developing: Edit code and it's immediately available (editable install)

## Common Commands

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=torchtune_config_writer

# Run specific test file
pytest tests/test_builder.py

# Install just core dependencies (no dev tools)
uv pip install -e .
```

## Troubleshooting

**Q: `uv: command not found` after running setup.sh**
A: The script installs uv but you need to add it to PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
# Then re-run the setup script or continue manually
```

**Q: `tune` command not found**
A: Make sure your virtual environment is activated: `source .venv/bin/activate`

**Q: `ModuleNotFoundError: No module named 'torchao'`**
A: Install torchao explicitly (it's now in dependencies, but if needed):
```bash
source .venv/bin/activate
uv pip install torchao
```

**Q: Build error: "Unable to determine which files to ship inside the wheel"**
A: The package directory structure doesn't exist. Create it:
```bash
mkdir -p torchtune_config_writer tests/fixtures
cat > torchtune_config_writer/__init__.py << 'EOF'
"""Torchtune Config Writer - Tool for generating torchtune YAML configs."""
__version__ = "0.1.0"
EOF
```

**Q: Import errors when running Python**
A: Ensure you're running Python from within the activated virtual environment

**Q: Want to use a specific torchtune version**
A: Edit `pyproject.toml` and change the version constraint, then run `uv pip install -e ".[dev]"` again

**Q: Changes to code not taking effect**
A: If you installed in editable mode (`-e`), changes should be immediate. If not, reinstall: `uv pip install -e ".[dev]"`
