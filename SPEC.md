# Torchtune Config Writer - Technical Specification

**Version**: 0.1.0
**Status**: 🚧 In Development - Phase 1 (Core Builder MVP)
**Last Updated**: 2025-01-09

---

## Executive Summary

**What**: A Python tool for generating torchtune YAML configuration files programmatically from existing recipes with customizations.

**Why**: Researchers need repeatable, testable config generation for fine-tuning experiments without manual YAML editing.

**How**:
- Start from torchtune's 100+ proven recipe configs
- Modify via Python builder API with type-safe deep merge
- Separate scientific parameters (what to learn) from engineering parameters (GPU efficiency)
- Track provenance via metadata for reproducibility
- Validate with torchtune's authoritative `tune validate`

**Current Status**: Designing Phase 1 (Core Builder). See [Implementation Plan](#implementation-plan) for roadmap.

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [Target Use Case](#target-use-case)
3. [Core Design Decisions](#core-design-decisions)
4. [Architecture Overview](#architecture-overview)
5. [Usage Patterns](#usage-patterns)
6. [Implementation Plan](#implementation-plan)
7. [File Structure & Dependencies](#file-structure--dependencies)
8. [Success Criteria](#success-criteria)
9. [Future Extensions](#future-extensions)
10. [Appendices Index](#appendices-index)

---

## Project Goal

Build a tool for generating torchtune configuration files in a repeatable, testable manner. Users specify their experiment parameters, and the tool generates valid torchtune YAML configs based on torchtune's existing recipe configs.

## Target Use Case

**Primary User**: Researchers running fine-tuning experiments who need to:
- Start from proven torchtune recipe configs
- Customize specific parameters (datasets, hyperparameters, model sizes)
- Generate multiple configs for parameter sweeps
- Track changes from baseline for reproducibility
- Validate configs before running expensive training jobs

**Example Workflow**:
```python
# Experiment 1: baseline
builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
builder.with_dataset("data/my_data.json")
builder.with_output_dir("results/exp_001")
builder.save("configs/exp_001.yaml")
builder.validate()

# Experiment 2: higher learning rate
builder2 = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
builder2.with_dataset("data/my_data.json")
builder2.with_learning_rate(1e-3)
builder2.save("configs/exp_002.yaml")
```

---

## Core Design Decisions

### 1. Config Format
**Decision**: Python dicts internally, export to YAML via PyYAML

**Rationale**: Easier to work with Python data structures, PyYAML handles edge cases, standard approach used by Kubernetes clients, Ansible, etc.

---

### 2. Base Strategy
**Decision**: Start from torchtune's existing recipe configs, allow customization

**Rationale**: Torchtune ships 100+ proven configs maintained by their team. Don't reinvent the wheel (Practical principle). Our value-add is programmatic generation + change tracking.

---

### 3. Validation Strategy
**Decision**: Use torchtune's `tune validate` command exclusively

**Rationale**: Authoritative validation from torchtune, checks YAML structure and component instantiation, single source of truth (Scientific principle), no redundant validation layer to maintain.

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
2. `from_file(path)` - From user's existing YAML files
3. `from_dict(config)` - From dict for programmatic creation
4. `from_previous(path)` - From previously generated config with metadata

---

### 6. Metadata Strategy
**Decision**: Store metadata in separate `.meta.yaml` files

**Rationale**: Avoids polluting configs with extra fields, maintains compatibility with `tune validate`, enables reproducibility, clean separation of concerns. See [Appendix D](appendices/D_metadata.md) for format details.

**Basic metadata format**:
```yaml
source_type: "torchtune_shipped"
base_config: "llama3_1/8B_lora_single_device"
tool_version: "0.1.0"
torchtune_version: "0.6.1"
generated_at: "2025-01-08T14:30:00Z"
overrides:
  optimizer:
    lr: 0.001
```

---

### 7. Scientific vs Engineering Parameters
**Decision**: Separate scientific decisions (what to learn) from engineering decisions (how to fit in memory)

**Rationale**: Researchers should focus on science, not GPU memory tuning. See [Appendix F](appendices/F_gpu_efficiency.md) for GPU helper specification.

**Scientific parameters** (affect experiment outcomes):
- Learning rate, epochs, weight decay, gradient accumulation
- Model architecture (LoRA rank, alpha, dropout)
- Data and loss configuration
- LR scheduling and optimizer settings

**Engineering parameters** (affect resources, not results):
- Batch size (GPU memory constraint)
- Activation checkpointing (memory/compute tradeoff)
- Device, dtype, compilation (hardware-specific)
- Output directories, logging, profiling

**Implementation**: Separate `GPUEfficiencyHelper` class manages engineering params and learns optimal settings over time (Self Improving principle).

---

## Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────┐
│          TorchtuneConfigBuilder                 │
│  (Scientific Parameters & Config Generation)    │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  Load Base Config                        │  │
│  │  - tune cp (torchtune shipped)           │  │
│  │  - from_file (user YAML)                 │  │
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

┌─────────────────────────────────────────────────┐
│          GPUEfficiencyHelper                    │
│  (Engineering Parameters & Self-Improving)      │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  Suggest Config                          │  │
│  │  - Bootstrap: heuristics                 │  │
│  │  - Learning: historical data             │  │
│  │  - Strategy: conservative/balanced/...   │  │
│  └──────────────────────────────────────────┘  │
│                    ↓                            │
│  ┌──────────────────────────────────────────┐  │
│  │  Learn from Runs                         │  │
│  │  - log_run_result()                      │  │
│  │  - parse_torchtune_log()                 │  │
│  │  - Improve over time                     │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Core Classes

#### TorchtuneConfigBuilder

Primary API for config generation. Handles scientific parameters.

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
delete(path: str) -> Self
with_dataset(path: str) -> Self
with_learning_rate(lr: float) -> Self
with_batch_size(batch_size: int) -> Self
# ... more with_*() methods

# Building & Validation
build() -> Dict
save(output_path: str, save_metadata: bool = True) -> str
validate(config_path: str = None) -> bool
```

**See**: [Appendix A](appendices/A_merge_semantics.md) for merge semantics, [Appendix B](appendices/B_config_loading.md) for loading details, [Appendix C](appendices/C_high_level_methods.md) for all high-level methods.

---

#### GPUEfficiencyHelper

Manages GPU-specific configuration and learns from runs.

**Key methods**:
```python
__init__(run_log_dir: str = None)
suggest_config(model_config: str, max_seq_length: int, strategy: str) -> Dict
log_run_result(config: Dict, result: Dict)
parse_torchtune_log(log_file: str) -> Dict
find_max_batch_size(model_config: str, max_seq_length: int, test_run: bool) -> int
```

**Evolution phases**:
1. **Bootstrap**: Uses heuristics and torchtune defaults
2. **Learning**: Accumulates data from your runs
3. **Optimization**: Provides smart recommendations

**See**: [Appendix F](appendices/F_gpu_efficiency.md) for complete specification.

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

### Pattern 2: Parameter Sweep

```python
for lr in [1e-4, 3e-4, 5e-4, 1e-3]:
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder.with_dataset("data/my_data.json")
    builder.with_learning_rate(lr)
    builder.with_output_dir(f"results/lr_{lr}")
    builder.save(f"configs/lr_{lr}.yaml")
    builder.validate()
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

### Pattern 4: Scientific + Engineering Separation

```python
# GPU helper handles engineering
gpu_helper = GPUEfficiencyHelper()
gpu_config = gpu_helper.suggest_config(
    model_config="llama3_1/8B_lora_single_device",
    max_seq_length=2048,
    strategy="balanced"
)

# Builder handles scientific parameters
builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
builder.override(gpu_config)  # Apply GPU settings
builder.with_learning_rate(3e-4)  # Focus on science
builder.with_dataset("data/my_data.json")
builder.save("configs/exp_001.yaml")

# After run completes, helper learns
run_result = GPUEfficiencyHelper.parse_torchtune_log("logs/exp_001.log")
gpu_helper.log_run_result(builder.build(), run_result)
```

---

### Pattern 5: Reproducible Experiment Specs

```python
# experiment_specs.py - version controlled
EXPERIMENTS = [
    {
        "base_config": "llama3_1/8B_lora_single_device",
        "overrides": {
            "dataset": {"source": "data/my_data.json"},
            "optimizer": {"lr": 3e-4},
            "output_dir": "results/baseline"
        }
    },
    {
        "base_config": "llama3_1/8B_lora_single_device",
        "overrides": {
            "dataset": {"source": "data/my_data.json"},
            "optimizer": {"lr": 1e-3},
            "output_dir": "results/high_lr"
        }
    }
]

# Generate all experiments
for spec in EXPERIMENTS:
    builder = TorchtuneConfigBuilder.from_dict(spec)
    builder.save(f"configs/{spec['overrides']['output_dir']}.yaml")
    builder.validate()
```

---

## Implementation Plan

### Phase 1: Core Builder (MVP) ⏳ Current

1. `TorchtuneConfigBuilder.__init__` - load base configs via `tune cp`
2. `override()` - deep merge implementation
3. `delete()` - delete keys by path
4. `build()` - apply operations in order
5. `save()` - write YAML (basic, no metadata)
6. High-level methods for scientific params: `with_dataset`, `with_learning_rate`, `with_epochs`
7. High-level method for engineering: `with_batch_size` (manual override)
8. Unit tests for core functionality

**Deliverables**: Working builder with basic config generation

---

### Phase 2: Validation & Metadata

1. `validate()` - call `tune validate` subprocess
2. `save()` with metadata - write `.meta.yaml`
3. `from_dict()` - construct from spec dict
4. `from_file()` - load from user YAML
5. Integration tests with `tune validate`

**Deliverables**: Validated configs with reproducibility metadata

---

### Phase 3: Discovery & Reuse

1. `list_available()` - discover torchtune configs via `tune ls`
2. `from_previous()` - load from metadata, reconstruct builder
3. Additional high-level methods based on usage (5 → 10-12 methods)
4. Documentation and examples

**Deliverables**: Full discovery and iteration workflow

---

### Phase 4: GPU Efficiency Helper (Self Improving)

1. `GPUEfficiencyHelper.__init__` - bootstrap with heuristics
2. `suggest_config()` - recommend GPU settings
3. Bootstrap knowledge: parse torchtune's shipped configs
4. `log_run_result()` - append to run_logs.jsonl
5. `parse_torchtune_log()` - extract metrics
6. Integration examples showing separation of concerns
7. Tests for heuristic and learning modes

**Deliverables**: Self-improving GPU optimization

---

### Phase 5: Batch Generation (Optional)

1. Separate `sweep_utils.py` module
2. Helper functions for common patterns
3. Examples for parameter sweeps
4. Integration with GPU helper

**Deliverables**: Tools for large-scale experiment generation

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
├── SESSION_LOG.md               # Development session notes
├── pyproject.toml               # Project config and dependencies
├── setup.sh                     # Automated setup script
│
├── appendices/                  # Detailed specifications
│   ├── A_merge_semantics.md
│   ├── B_config_loading.md
│   ├── C_high_level_methods.md (🚧 TODO)
│   ├── D_metadata.md (🚧 TODO)
│   ├── E_testing.md
│   └── F_gpu_efficiency.md (🚧 TODO)
│
├── config_builder.py            # TorchtuneConfigBuilder (Phase 1-3)
├── gpu_efficiency.py            # GPUEfficiencyHelper (Phase 4)
├── sweep_utils.py               # Batch generation helpers (Phase 5)
│
├── tests/                       # Test suite
│   ├── test_builder.py
│   ├── test_integration.py
│   ├── test_metadata.py
│   ├── test_gpu_efficiency.py
│   └── fixtures/
│
├── examples/                    # Usage examples
│   ├── basic_usage.py
│   ├── parameter_sweep.py
│   ├── gpu_efficiency_bootstrap.py
│   ├── gpu_efficiency_learning.py
│   └── experiment_tracking.py
│
├── example_configs/             # Sample configs for development
│   ├── llama3_1_8B_lora.yaml
│   ├── llama3_1_8B_full.yaml
│   └── llama3_2_1B_lora.yaml
│
├── logs/                        # Run logs (JSONL, gitignored)
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
2. **Usability**: Simple API for common operations, clear error messages
3. **Reproducibility**: Metadata enables recreation of any config
4. **Testability**: Comprehensive test coverage of core functionality
5. **Maintainability**: Clean, modular code that's easy to extend
6. **Documentation**: Clear examples for common use cases

---

## Future Extensions (Post-MVP)

1. **More high-level methods**: Add based on actual usage patterns
2. **Config diff tool**: Compare two configs, show differences
3. **Template library**: Save user's common patterns as reusable templates
4. **Validation hooks**: Custom validation rules for specific use cases
5. **Multi-config management**: Tools for managing experiment suites
6. **Experiment tracking integration**: Export to MLflow, Weights & Biases
7. **GPU efficiency improvements**:
   - Automatic log parsing and ingestion
   - Binary search for max batch size via test runs
   - Community knowledge sharing (opt-in, anonymized)
   - Advanced memory estimation models
   - Multi-GPU and distributed training support

---

## Alignment with Guiding Principles

**Scientific**:
- Metadata for reproducibility, authoritative validation, change tracking
- Clear separation of scientific vs engineering parameters
- Researchers focus on what matters: learning rates, architectures, data

**Modular**:
- Separate concerns (builder, metadata, validation, GPU efficiency)
- Scientific decisions in TorchtuneConfigBuilder
- Engineering decisions in GPUEfficiencyHelper
- Easy to extend and compose

**Practical**:
- Leverage existing torchtune configs, simple implementation
- No over-engineering: start with heuristics, improve over time
- GPU helper works day 1, gets better with use

**Privacy Respecting**:
- No data collection, all local processing
- Run logs stored locally in JSONL format
- Future community sharing is opt-in and anonymized

**Self Improving**:
- GPU helper learns from every run
- Bootstrap → Learning → Optimization evolution
- Track successful configs, expand API based on usage
- Continuous improvement loop built into design

**Tested**:
- Comprehensive tests, validation at every step
- Tests for both heuristic and learning modes
- Integration tests with actual torchtune validation

See `CLAUDE.md` for complete guiding principles.

---

## Appendices Index

Detailed specifications are in separate appendices for maintainability:

- **[Appendix A: Config Merge Semantics](appendices/A_merge_semantics.md)** ✅
  Complete specification of `override()` and `delete()` behavior, including merge rules, type compatibility, edge cases, and implementation details.

- **[Appendix B: Config Loading Strategy](appendices/B_config_loading.md)** ✅
  Four loading methods (torchtune shipped, file, dict, previous), source tracking, path resolution, and error handling.

- **[Appendix C: High-Level Methods](appendices/C_high_level_methods.md)** ✅
  Complete specification of all `with_*()` convenience methods, organized by scientific vs engineering parameters.

- **[Appendix D: Metadata Format](appendices/D_metadata.md)** ✅
  Metadata structure, environment tracking, data provenance, privacy considerations, and file format decisions.

- **[Appendix E: Testing Strategy](appendices/E_testing.md)** ✅
  Comprehensive testing approach including unit tests, integration tests, test organization, and coverage goals.

- **[Appendix F: GPU Efficiency Helper](appendices/F_gpu_efficiency.md)** ✅
  Complete specification of GPUEfficiencyHelper class, bootstrap heuristics, learning algorithms, and integration patterns.

---

**Legend**: ✅ Complete | 🚧 TODO/In Progress
