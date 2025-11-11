# Torchtune Config Writer

A tool for generating torchtune YAML configuration files programmatically from existing recipes with customizations.

Researchers running fine-tuning experiments often need to start from proven torchtune recipe configs, customize specific parameters (datasets, hyperparameters, model sizes), generate multiple configs for parameter sweeps, and validate configs before running expensive training jobs. This tool provides a Python API to do exactly that.

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

## Quick Start

```bash
./setup.sh
source .venv/bin/activate
pytest  # verify installation
```

See [SETUP.md](SETUP.md) for details.

## Documentation

- **[SETUP.md](SETUP.md)** - Environment setup
- **[SPEC.md](SPEC.md)** - Technical specification
- **[CLAUDE.md](CLAUDE.md)** - Development principles
