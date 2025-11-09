# Torchtune Config Writer - Technical Specification

**Version**: 0.2.0
**Status**: 🚧 In Development - Phase 1 (Core Builder MVP)
**Last Updated**: 2025-01-09

---

## Executive Summary

**What**: A Python tool for generating torchtune YAML configuration files programmatically from existing recipes with customizations.

**Why**: Researchers running SLURM-based training jobs need repeatable, testable config generation for parameter sweeps without manual YAML editing.

**How**:
- Start from torchtune's 100+ proven recipe configs (already optimized for A100-80GB)
- Modify via Python builder API with type-safe deep merge
- Generate multiple configs for parameter sweeps (submit to SLURM queue)
- Track provenance via metadata for reproducibility
- Validate with torchtune's authoritative `tune validate` (catch errors before queue submission)

**Current Status**: Designing Phase 1 (Core Builder). See [Implementation Plan](#implementation-plan) for roadmap.

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [Target Use Case](#target-use-case)
3. [Core Design Decisions](#core-design-decisions)
4. [Architecture Overview](#architecture-overview)
5. [Usage Patterns](#usage-patterns)
6. [SLURM Integration](#slurm-integration)
7. [Implementation Plan](#implementation-plan)
8. [File Structure & Dependencies](#file-structure--dependencies)
9. [Success Criteria](#success-criteria)
10. [Appendices Index](#appendices-index)

---

## Project Goal

Build a tool for generating torchtune configuration files for SLURM-based training jobs in a repeatable, testable manner. Users specify their experiment parameters, and the tool generates valid torchtune YAML configs based on torchtune's existing recipe configs.

**Key differentiator**: Batch generation for parameter sweeps. Generate and validate 100 configs in seconds, then submit to SLURM queue.

## Target Use Case

**Primary User**: Researchers running fine-tuning experiments on SLURM clusters who need to:
- Start from proven torchtune recipe configs (already GPU-optimized)
- Generate multiple configs for parameter sweeps efficiently
- Validate configs before submitting to queue (avoid wasting allocation time)
- Track changes from baseline for reproducibility
- Adapt configs for different GPU types when needed

**Primary Environment**: SLURM cluster with GPU nodes
- Jobs submitted via `sbatch` scripts
- GPU type explicit in SLURM allocation (`#SBATCH --gres=gpu:a100:1`)
- Cannot iterate quickly (queue wait times)
- Want configs that work first try (OOM wastes precious allocation time)

**Example Workflow**:
```python
# Generate sweep of 16 configs in seconds
import itertools

learning_rates = [1e-4, 3e-4, 5e-4, 1e-3]
lora_ranks = [8, 16, 32, 64]

for lr, rank in itertools.product(learning_rates, lora_ranks):
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder.with_learning_rate(lr)
    builder.with_lora_params(rank, rank * 2)
    builder.with_dataset("data/my_data.json")
    builder.with_output_dir(f"results/lr_{lr}_rank_{rank}")
    builder.save(f"configs/lr_{lr}_rank_{rank}.yaml")
    builder.validate()  # Catch errors before submitting to queue!

# Submit all 16 jobs to SLURM
# sbatch --array=0-15 run_sweep.sh
```

**Why this matters for SLURM**:
- ✅ Generate many configs quickly (not manually editing YAML)
- ✅ Validate before queue submission (catch errors early)
- ✅ Reproducible (metadata tracks what changed)
- ✅ GPU settings from torchtune configs already work (no tuning needed)

---

## Core Design Decisions

### 1. Config Format
**Decision**: Python dicts internally, export to YAML via PyYAML

**Rationale**: Easier to work with Python data structures, PyYAML handles edge cases, standard approach used by Kubernetes clients, Ansible, etc.

---

### 2. Base Strategy
**Decision**: Start from torchtune's existing recipe configs, allow customization

**Rationale**: Torchtune ships 100+ proven configs maintained by their team, already optimized for common GPUs (A100-80GB). Don't reinvent the wheel (Practical principle). Our value-add is programmatic generation + change tracking for SLURM parameter sweeps.

**Key insight**: Torchtune's shipped configs have good batch_size, dtype, and checkpointing defaults. Users can manually adjust once if using different GPU, then generate sweep from that base.

---

### 3. Validation Strategy
**Decision**: Use torchtune's `tune validate` command exclusively

**Rationale**: Authoritative validation from torchtune, checks YAML structure and component instantiation, single source of truth (Scientific principle), no redundant validation layer to maintain. **Critical for SLURM**: catch config errors before submitting to queue.

---

### 4. Modification Approach
**Decision**: Nested dict merge with high-level helpers

**Rationale**: Simple, Pythonic, flexible. Deep merge allows overriding any nested structure. High-level methods for common operations (80/20 rule). See [Appendix A](appendices/A_merge_semantics.md) for complete merge semantics.

**Key merge rules:**
- **Scalars** (str, int, float, bool, None) → Replace
- **Lists** → Replace entirely (no extending)
- **Dicts** → Deep merge (recursive)
- **Type safety**: Compatible types only
- **Deletion**: Explicit `delete()` method

---

### 5. Config Loading Strategy
**Decision**: Four loading methods to support different workflows

**Rationale**: Flexibility for different use cases while maintaining provenance tracking. See [Appendix B](appendices/B_config_loading.md) for complete specification.

**Loading methods:**
1. `TorchtuneConfigBuilder(name)` - From torchtune's shipped configs via `tune cp`
2. `from_file(path)` - From user's existing YAML files (e.g., GPU-adapted base config)
3. `from_dict(config)` - From dict for programmatic creation
4. `from_previous(path)` - From previously generated config with metadata

**SLURM workflow**: Load from torchtune config OR custom GPU-adapted base, then generate parameter sweep.

---

### 6. Metadata Strategy
**Decision**: Store metadata in separate `.meta.yaml` files

**Rationale**: Avoids polluting configs with extra fields, maintains compatibility with `tune validate`, enables reproducibility, clean separation of concerns. See [Appendix D](appendices/D_metadata.md) for format details.

**Basic metadata format**:
```yaml
source_type: "torchtune_shipped"
base_config: "llama3_1/8B_lora_single_device"
tool_version: "0.2.0"
torchtune_version: "0.6.1"
generated_at: "2025-01-09T14:30:00Z"
overrides:
  optimizer:
    lr: 0.001
```

---

## Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────┐
│          TorchtuneConfigBuilder                 │
│      (Config Generation for SLURM Jobs)         │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  Load Base Config                        │  │
│  │  - tune cp (torchtune shipped)           │  │
│  │  - from_file (GPU-adapted base)          │  │
│  │  - from_dict (programmatic)              │  │
│  │  - from_previous (with metadata)         │  │
│  └──────────────────────────────────────────┘  │
│                    ↓                            │
│  ┌──────────────────────────────────────────┐  │
│  │  Apply Modifications                     │  │
│  │  - override() (deep merge)               │  │
│  │  - delete() (remove keys)                │  │
│  │  - with_*() methods (high-level)         │  │
│  └──────────────────────────────────────────┘  │
│                    ↓                            │
│  ┌──────────────────────────────────────────┐  │
│  │  Build & Save                            │  │
│  │  - build() → Dict                        │  │
│  │  - save() → YAML + metadata              │  │
│  │  - validate() → tune validate            │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Core Class: TorchtuneConfigBuilder

Primary API for config generation. Single-component architecture.

**Key methods**:
```python
# Loading
__init__(base_config_name: str)
from_file(config_path: str)
from_dict(config: Dict)
from_previous(config_path: str)
list_available() -> List[str]

# Modification
override(updates: Dict) -> Self
with_dataset(path: str) -> Self
with_output_dir(path: str) -> Self
with_learning_rate(lr: float) -> Self
with_epochs(epochs: int) -> Self
with_lora_params(rank: int, alpha: int) -> Self
with_batch_size(batch_size: int) -> Self
with_packed(enabled: bool) -> Self
with_dataset_template(template: str) -> Self
with_seed(seed: int) -> Self
# Note: delete() deferred to Phase 2

# Building & Validation
build() -> Dict
save(output_path: str, save_metadata: bool = True) -> str
validate(config_path: str = None) -> bool
```

**See**:
- [Appendix A](appendices/A_merge_semantics.md) for merge semantics
- [Appendix B](appendices/B_config_loading.md) for loading details
- [Appendix C](appendices/C_high_level_methods.md) for all high-level methods

---

## Usage Patterns

### Pattern 1: Simple Config Generation

```python
builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
builder.with_dataset("data/my_data.json")
builder.with_learning_rate(3e-4)
builder.with_output_dir("results/exp_001")
builder.save("configs/exp_001.yaml")
builder.validate()
```

---

### Pattern 2: Parameter Sweep for SLURM

```python
# Generate multiple configs for SLURM job array
for lr in [1e-4, 3e-4, 5e-4, 1e-3]:
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder.with_dataset("data/my_data.json")
    builder.with_learning_rate(lr)
    builder.with_output_dir(f"results/lr_{lr}")
    builder.save(f"configs/lr_{lr}.yaml")
    builder.validate()

# All configs validated before queue submission!
```

---

### Pattern 3: Build on Previous Success

```python
# Load successful experiment
builder = TorchtuneConfigBuilder.from_previous("configs/exp_042.yaml")

# Tweak one parameter
builder.with_learning_rate(1e-3)

# Save as new experiment
builder.save("configs/exp_043.yaml")
```

---

### Pattern 4: Adapt for Different GPU (One-Time Setup)

```python
# Torchtune configs assume A100-80GB
# If using V100-16GB, adapt once:

builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
# Reduce for smaller GPU
builder.with_batch_size(1)  # was 2
builder.with_activation_checkpointing(True)  # was False
builder.save("configs/v100_base.yaml")

# Now generate sweep from adapted base
for lr in learning_rates:
    builder = TorchtuneConfigBuilder.from_file("configs/v100_base.yaml")
    builder.with_learning_rate(lr)
    builder.save(f"configs/v100_lr_{lr}.yaml")
    builder.validate()
```

---

### Pattern 5: Multi-Dimensional Sweep

```python
# Generate full hyperparameter grid
import itertools

learning_rates = [1e-4, 3e-4, 5e-4, 1e-3]
lora_ranks = [8, 16, 32, 64]
weight_decays = [0.0, 0.01, 0.1]

# 4 x 4 x 3 = 48 configs
for lr, rank, wd in itertools.product(learning_rates, lora_ranks, weight_decays):
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder.with_learning_rate(lr)
    builder.with_lora_params(rank, rank * 2)
    builder.override({"optimizer": {"weight_decay": wd}})
    builder.with_output_dir(f"results/lr_{lr}_rank_{rank}_wd_{wd}")

    config_name = f"lr_{lr}_rank_{rank}_wd_{wd}"
    builder.save(f"configs/{config_name}.yaml")
    builder.validate()

print(f"Generated 48 configs, all validated!")
```

---

## SLURM Integration

### Why This Tool is Perfect for SLURM

**SLURM challenges**:
- ❌ Can't iterate quickly (queue wait times)
- ❌ Need configs that work first try (OOM wastes allocation)
- ❌ Generating many configs manually is error-prone
- ❌ Hard to track what changed between experiments

**How this tool helps**:
- ✅ Generate 100+ configs in seconds
- ✅ Validate before queue submission (catch errors early)
- ✅ Start from proven GPU-optimized configs
- ✅ Reproducibility via metadata

---

### SLURM Workflow Pattern

```python
# 1. Generate configs
import itertools

learning_rates = [1e-4, 3e-4, 5e-4, 1e-3]
lora_ranks = [8, 16, 32, 64]

configs_generated = []

for lr, rank in itertools.product(learning_rates, lora_ranks):
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder.with_learning_rate(lr)
    builder.with_lora_params(rank, rank * 2)
    builder.with_dataset("/shared/data/my_dataset.json")
    builder.with_output_dir(f"/scratch/results/lr_{lr}_rank_{rank}")

    config_name = f"lr_{lr}_rank_{rank}"
    config_path = f"configs/{config_name}.yaml"
    builder.save(config_path)
    builder.validate()

    configs_generated.append(config_path)

print(f"✓ Generated and validated {len(configs_generated)} configs")

# 2. Generate SLURM job array script
slurm_script = f"""#!/bin/bash
#SBATCH --job-name=llama_sweep
#SBATCH --array=0-{len(configs_generated)-1}
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# Get config for this array task
CONFIGS=({' '.join(configs_generated)})
CONFIG=${{CONFIGS[$SLURM_ARRAY_TASK_ID]}}

echo "Running config: $CONFIG"
tune run lora_finetune_single_device --config $CONFIG
"""

with open("run_sweep.sh", "w") as f:
    f.write(slurm_script)

print("✓ Generated SLURM script: run_sweep.sh")
print(f"Submit with: sbatch run_sweep.sh")
```

---

### SLURM Script Generation (Phase 3)

**Approach**: Template-based with Claude Code skill for flexibility

For Phase 3, we'll provide SLURM script templates and a Claude Code skill for interactive generation, rather than a Python utility. This approach is more flexible for different cluster configurations.

**Option 1: Copy-paste template** (simplest):

```bash
#!/bin/bash
#SBATCH --job-name=torchtune_sweep
#SBATCH --array=0-15  # Adjust for your number of configs
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# List your config files
CONFIGS=(
    configs/lr_0.0001.yaml
    configs/lr_0.0003.yaml
    # ... add all your configs
)

# Get config for this array task
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Running config: $CONFIG"
tune run lora_finetune_single_device --config $CONFIG
```

**Option 2: Claude Code skill** (Phase 3, more flexible):

A Claude Code skill will interactively generate SLURM scripts by:
- Reading your generated configs
- Asking about cluster parameters (GPU type, partition, time limits)
- Generating customized SLURM scripts
- Adapting to your specific cluster configuration

This provides more flexibility than a hardcoded Python utility while still being easy to use.

---

### Real-World SLURM Example

Complete workflow from config generation to job submission:

```python
#!/usr/bin/env python3
"""
generate_sweep.py - Generate configs and SLURM script for LR sweep
"""

from torchtune_config_writer import TorchtuneConfigBuilder
from torchtune_config_writer.slurm_utils import generate_slurm_array_script
import os

# Create directories
os.makedirs("configs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Parameter sweep
learning_rates = [1e-4, 3e-4, 5e-4, 1e-3]
config_paths = []

print("Generating configs...")
for lr in learning_rates:
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")

    # Scientific parameters
    builder.with_learning_rate(lr)
    builder.with_lora_params(32, 64)
    builder.with_epochs(3)
    builder.with_dataset("/shared/datasets/alpaca_cleaned.json")
    builder.with_output_dir(f"/scratch/$USER/results/lr_{lr}")

    # Save and validate
    config_path = f"configs/lr_{lr}.yaml"
    builder.save(config_path)

    try:
        builder.validate()
        print(f"  ✓ {config_path}")
        config_paths.append(config_path)
    except Exception as e:
        print(f"  ✗ {config_path}: {e}")

# Generate SLURM script
if config_paths:
    generate_slurm_array_script(
        config_paths=config_paths,
        output_path="run_sweep.sh",
        job_name="llama3_lr_sweep",
        gpu_type="a100",
        mem_gb=64,
        time_hours=12
    )

    print(f"\n✓ Generated {len(config_paths)} configs")
    print(f"✓ Generated run_sweep.sh")
    print(f"\nSubmit with: sbatch run_sweep.sh")
else:
    print("\n✗ No valid configs generated!")
```

**Run**:
```bash
$ python generate_sweep.py
Generating configs...
  ✓ configs/lr_0.0001.yaml
  ✓ configs/lr_0.0003.yaml
  ✓ configs/lr_0.0005.yaml
  ✓ configs/lr_0.001.yaml

✓ Generated 4 configs
✓ Generated run_sweep.sh

Submit with: sbatch run_sweep.sh

$ sbatch run_sweep.sh
Submitted batch job 123456
```

---

## Implementation Plan

### Phase 1: Core Builder (MVP) ⏳ Current

**Duration**: 3-4 weeks

**Components**:
1. `TorchtuneConfigBuilder.__init__` - load base configs via `tune cp`
2. `override()` - deep merge implementation
3. `build()` - apply operations in order
4. `save()` - write YAML with simplified metadata
5. `validate()` - call `tune validate` subprocess
6. Essential high-level methods (9 methods):
   - `with_dataset()` - Change dataset path
   - `with_output_dir()` - Change output directory
   - `with_learning_rate()` - Change optimizer LR
   - `with_epochs()` - Change training duration
   - `with_lora_params()` - Change LoRA architecture (rank + alpha paired)
   - `with_batch_size()` - Manual GPU memory control
   - `with_packed()` - Enable/disable sequence packing
   - `with_dataset_template()` - Set prompt template for SFT
   - `with_seed()` - Set random seed for reproducibility
7. Simplified metadata (source, versions, timestamp, overrides only)
8. Unit tests for core functionality

**Deliverables**: Working builder with config generation, validation, and simplified metadata

**SLURM value**: Generate and validate multiple configs before queue submission

**Note**: `delete()` method deferred to Phase 2

---

### Phase 2: Discovery & Reuse

**Duration**: 1-2 weeks

**Components**:
1. `list_available()` - discover torchtune configs via `tune ls`
2. `from_file()` - load from user YAML (e.g., GPU-adapted base)
3. `from_dict()` - construct from spec dict
4. `from_previous()` - load from metadata, reconstruct builder
5. `delete()` - delete keys by path (deferred from Phase 1)
6. Enhanced metadata with privacy features (path sanitization, environment tracking)
7. Additional high-level methods based on usage (2-4 more methods as needed)
8. Integration tests with `tune validate`
9. Documentation and examples

**Deliverables**: Full discovery and iteration workflow with metadata roundtrip

**SLURM value**: Adapt configs for different GPUs, iterate on successful experiments

---

### Phase 3: SLURM Integration & Batch Generation

**Duration**: 1-2 days

**Components**:
1. SLURM script templates for common patterns (job arrays, parameter sweeps)
2. Claude Code skill for interactive SLURM script generation
3. Examples for common SLURM workflows
4. Documentation for SLURM integration

**Deliverables**: SLURM templates and skill for flexible script generation

**SLURM value**: End-to-end workflow from config generation to job submission

**Note**: Skill-based approach provides more flexibility than hardcoded Python utility, adapts to different cluster configurations

---

### Total Timeline: 4-6 weeks

- Phase 1: 3-4 weeks
- Phase 2: 1-2 weeks
- Phase 3: 1-2 days

Much faster than original 8-12 weeks by removing GPU helper complexity and using skill-based SLURM approach.

---

## File Structure & Dependencies

### Project Structure

```
torchtune_config_writer/
├── .venv/                       # Virtual environment (gitignored)
├── .gitignore                   # Git ignore rules
├── CLAUDE.md                    # Guiding principles
├── SPEC.md                      # This document (master spec)
├── SETUP.md                     # Environment setup instructions
├── README.md                    # Project overview
├── pyproject.toml               # Project config and dependencies
├── setup.sh                     # Automated setup script
│
├── appendices/                  # Detailed specifications
│   ├── A_merge_semantics.md     # ✅ Complete
│   ├── B_config_loading.md      # ✅ Complete
│   ├── C_high_level_methods.md  # ✅ Complete
│   ├── D_metadata.md            # ✅ Complete
│   └── E_testing.md             # ✅ Complete
│
├── config_builder.py            # TorchtuneConfigBuilder
├── slurm_templates/             # SLURM script templates (Phase 3)
│   ├── job_array_basic.sh       # Basic job array template
│   └── job_array_sweep.sh       # Parameter sweep template
│
├── tests/                       # Test suite
│   ├── test_builder.py          # Core builder tests
│   ├── test_integration.py      # Integration with tune validate
│   ├── test_metadata.py         # Metadata generation tests
│   └── fixtures/                # Test fixtures
│
├── examples/                    # Usage examples
│   ├── basic_usage.py
│   ├── parameter_sweep.py
│   ├── slurm_integration.py     # SLURM workflow example
│   ├── multi_gpu_adaptation.py  # Adapting for different GPUs
│   └── experiment_tracking.py
│
├── example_configs/             # Sample configs
│   ├── llama3_1_8B_lora.yaml
│   ├── llama3_1_8B_full.yaml
│   └── llama3_2_1B_lora.yaml
│
└── scratch/                     # Temporary experiments (gitignored)
```

### Dependencies

All dependencies specified in `pyproject.toml`:

**Core dependencies**:
- `torchtune>=0.3.0,<1.0.0` - Base configs and validation
- `torchao>=0.14.0` - Required by torchtune
- `pyyaml>=6.0,<7.0` - YAML generation

**Development dependencies**:
- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage reporting

**Installation**:
```bash
# Automated setup (recommended)
./setup.sh

# Manual installation
uv pip install -e ".[dev]"
```

See `SETUP.md` for detailed installation instructions.

---

## Success Criteria

1. **Correctness**: All generated configs pass `tune validate`
2. **Efficiency**: Generate 100 configs in < 10 seconds
3. **Usability**: Simple API for common operations, clear error messages
4. **Reproducibility**: Metadata enables recreation of any config
5. **SLURM Integration**: Seamless workflow from config generation to job submission
6. **Testability**: Comprehensive test coverage of core functionality
7. **Maintainability**: Clean, modular code that's easy to extend
8. **Documentation**: Clear examples for SLURM use cases

---

## Alignment with Guiding Principles

**Scientific**:
- Metadata for reproducibility, authoritative validation, change tracking
- Validate configs before queue submission (avoid wasting allocation time)
- Start from proven torchtune configs (A100-optimized)

**Modular**:
- Single-component architecture (TorchtuneConfigBuilder)
- Clear separation of core builder vs optional utilities
- Easy to extend with new high-level methods

**Practical**:
- Leverage existing torchtune configs (don't reinvent GPU tuning)
- Focus on real use case: SLURM parameter sweeps
- Simple implementation: no complex learning or optimization
- Works with existing SLURM infrastructure

**Privacy Respecting**:
- No data collection, all local processing
- Metadata stored locally with configs
- No external dependencies or services

**Self Improving**:
- Track successful configs via metadata
- Add high-level methods based on actual usage patterns
- Community can share GPU-adapted base configs

**Tested**:
- Comprehensive tests for core functionality
- Integration tests with actual torchtune validation
- Test SLURM utilities with realistic examples

See `CLAUDE.md` for complete guiding principles.

---

## Appendices Index

Detailed specifications are in separate appendices for maintainability:

- **[Appendix A: Config Merge Semantics](appendices/A_merge_semantics.md)** ✅
  Complete specification of `override()` and `delete()` behavior, including merge rules, type compatibility, edge cases, and implementation details.

- **[Appendix B: Config Loading Strategy](appendices/B_config_loading.md)** ✅
  Four loading methods (torchtune shipped, file, dict, previous), source tracking, path resolution, and error handling.

- **[Appendix C: High-Level Methods](appendices/C_high_level_methods.md)** ✅
  Complete specification of all `with_*()` convenience methods. Focus on 6-7 essential methods for Phase 1.

- **[Appendix D: Metadata Format](appendices/D_metadata.md)** ✅
  Metadata structure, environment tracking, data provenance, privacy considerations, and file format decisions. Focus on basic metadata for Phase 1.

- **[Appendix E: Testing Strategy](appendices/E_testing.md)** ✅
  Comprehensive testing approach including unit tests, integration tests, test organization, and coverage goals.

---

## Changes from v0.1.0

**Major changes**:
1. **Added SLURM context** - Primary use case is now SLURM-based training
2. **Removed GPU Efficiency Helper** - Appendix F deleted, Phase 4 removed
3. **Simplified architecture** - Single-component design (TorchtuneConfigBuilder only)
4. **Added SLURM integration** - New section with utilities for job array generation
5. **Reduced scope** - 5-7 weeks instead of 8-12 weeks

**Rationale**:
- Torchtune configs already have good GPU defaults
- SLURM users know their GPU type (explicit in allocation)
- Can't iterate quickly on SLURM (queue wait times)
- Real value is batch config generation, not GPU tuning
- Focus on practical use case: parameter sweeps for SLURM clusters

---

**Legend**: ✅ Complete | 🚧 TODO/In Progress
