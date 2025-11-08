# Torchtune Config Writer

A tool for generating torchtune YAML configuration files programmatically from existing recipes with customizations.

## Quick Start

```bash
# One-command setup
./setup.sh
source .venv/bin/activate

# Verify installation
tune ls
pytest
```

## What This Tool Does

Researchers running fine-tuning experiments often need to:
- Start from proven torchtune recipe configs
- Customize specific parameters (datasets, hyperparameters, model sizes)
- Generate multiple configs for parameter sweeps
- Track changes from baseline for reproducibility
- Validate configs before running expensive training jobs

This tool provides a Python API to do exactly that.

## Example Usage

```python
from torchtune_config_writer import TorchtuneConfigBuilder

# Create a config based on a torchtune recipe
builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
builder.with_dataset("data/my_data.json")
builder.with_learning_rate(1e-3)
builder.with_output_dir("results/exp_001")

# Save and validate
builder.save("configs/exp_001.yaml")
builder.validate()
```

## Documentation

- **[SETUP.md](SETUP.md)** - Environment setup instructions
- **[SPEC.md](SPEC.md)** - Complete technical specification and API design
- **[CLAUDE.md](CLAUDE.md)** - Guiding principles for development

## Development Setup

```bash
# Automated setup (recommended)
./setup.sh
source .venv/bin/activate

# Manual setup
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=torchtune_config_writer

# Run specific test file
pytest tests/test_builder.py
```

## Project Status

🚧 **In Development** - Currently implementing Phase 1 (Core Builder MVP)

See [SPEC.md](SPEC.md) for the complete implementation plan.

## Dependencies

- Python 3.10+
- torchtune (for base configs and validation)
- PyYAML (for config generation)

See `pyproject.toml` for the complete dependency specification.

## Guiding Principles

- **Scientific**: Emphasize correctness, reproducibility, and detailed logging
- **Modular**: Design components to be added or changed with minimal impact
- **Practical**: Do science, not programming contests. Don't over-engineer
- **Privacy Respecting**: All processing is local, no data collection
- **Self Improving**: Learn from experiments to improve future work
- **Tested**: Write tests alongside development

See [CLAUDE.md](CLAUDE.md) for more details.

## License

MIT
