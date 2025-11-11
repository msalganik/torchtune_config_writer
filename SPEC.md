# Torchtune Config Writer - Technical Specification

**Version**: 0.2.0 (Phase 1)
**Status**: 🚧 Implementation in Progress
**Last Updated**: 2025-11-11

---

## Executive Summary

**What**: A Python tool for generating multiple torchtune YAML config files from parameter sweeps defined in `experiment.yaml`.

**Why**: Researchers need to run parameter sweeps on SLURM clusters. Manually creating and editing dozens of YAML configs is error-prone and doesn't reflect scientific thinking (variables vs. controls).

**How**:
1. Define experiment in `experiment.yaml` with **variables** (what sweeps) and **controls** (what's constant)
2. Tool generates complete torchtune config for each variable combination
3. Start from torchtune's proven recipes or custom configs
4. Validate all configs before submitting to SLURM queue

**Example:**
```yaml
# experiment.yaml
experiment_name: lora_rank_study
base_config: llama3_2/1B_lora_single_device

variables:
  lora_rank: [8, 16, 32, 64]  # 4 values = 4 configs

controls:
  learning_rate: 3e-4
  dataset:
    data_files: /path/to/data.json
```

Generates 4 complete torchtune configs, ready to submit to SLURM.

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [Core Concept](#core-concept)
3. [Design Decisions](#design-decisions)
4. [Architecture](#architecture)
5. [Phase 1 Scope](#phase-1-scope)
6. [Usage](#usage)
7. [Implementation Plan](#implementation-plan)
8. [File Structure](#file-structure)
9. [Appendices](#appendices)

---

## Project Goal

Enable researchers to define fine-tuning experiments in terms they think in (variables and controls) and automatically generate valid torchtune configs for SLURM parameter sweeps.

**Not a goal**: Replace torchtune's config system. We use their configs as a starting point.

---

## Core Concept

### The Problem

Researchers think in terms of:
- **Independent variables** (what I'm testing): lora_rank = [8, 16, 32, 64]
- **Controlled variables** (what stays the same): learning_rate = 3e-4

But torchtune needs separate YAML files for each configuration. Manually creating/editing these is:
- Time-consuming (copy-paste errors)
- Error-prone (forget to change a field)
- Not reviewable (hard to see what actually varies)
- Not testable

### The Solution

Define experiments declaratively:

```yaml
variables:   # What I'm testing
  lora_rank: [8, 16, 32]

controls:    # What stays constant
  learning_rate: 3e-4
  epochs: 2
```

Tool generates cartesian product → N complete torchtune configs.

**Scientific benefit**: experiment.yaml shows your experimental design clearly. Six months later, you can see exactly what varied and what didn't.

---

## Design Decisions

### 1. Base Config Strategy

**Decision**: Start from torchtune's existing recipe configs OR user's custom config file

**Rationale**:
- Torchtune ships 100+ proven configs already optimized for A100-80GB GPUs
- Users can adapt one config for their GPU, then generate sweeps from it
- Don't reinvent defaults - use torchtune's expertise

**Implementation**:
```yaml
# Option 1: Use torchtune recipe
base_config: "llama3_2/1B_lora_single_device"

# Option 2: Use custom file
base_config_file: "/path/to/my_custom_config.yaml"
```

Tool loads base, then applies overrides.

---

### 2. Variables vs Controls

**Decision**: Explicit separation in experiment.yaml

**Rationale**:
- Mirrors scientific experimental design
- Makes it obvious what's being tested
- Easier to review experiment plans
- Clear documentation for papers/repos

**Alternative considered**: Flat parameter list with metadata tags. Rejected as less clear.

---

### 3. Merge Strategy

**Decision**: Deep merge for nested dict overrides, replace for scalars/lists

**Rationale**:
- Scalar replace: `learning_rate: 3e-4` just sets the value
- Nested merge: Can override `dataset.data_files` without rewriting entire dataset section
- List replace: Less surprising than append/merge

See [Appendix A](appendices/A_merge_semantics.md) for complete merge rules.

---

### 4. Validation Strategy

**Decision**: Use `tune validate` for generated configs (optional but recommended)

**Rationale**:
- Authoritative validation from torchtune itself
- Catches component instantiation errors
- **Critical for SLURM**: Find config errors before queue submission

**Implementation**: After generating all configs, optionally run `tune validate` on each.

---

### 5. No Builder API in Phase 1

**Decision**: Phase 1 is YAML → configs only, no Python fluent API

**Rationale**:
- Simple, focused implementation
- experiment.yaml is the interface
- Can add Python API later if needed (for programmatic sweep generation)

---

## Architecture

### Data Flow

```
experiment.yaml
      ↓
   [Parser]
      ↓
   Pydantic Models (validation)
      ↓
   [Base Config Loader]
      ├─ tune cp (for torchtune recipes)
      └─ Read file (for custom configs)
      ↓
   Base Config Dict
      ↓
   [Merge Engine]
      ├─ Apply controls (to all)
      └─ Apply each variable combination
      ↓
   N Config Dicts (one per variable combo)
      ↓
   [YAML Writer]
      ↓
   configs/
      ├─ run_000.yaml
      ├─ run_001.yaml
      ├─ ...
      ├─ run_NNN.yaml
      └─ run_mapping.yaml
```

### Core Components

**1. Schema (Pydantic Models)**
- `ExperimentConfig`: Top-level experiment definition
- `VariablesConfig`: Parameters to sweep
- `ControlsConfig`: Fixed parameters
- Validation: required fields, type checking, constraints

**2. Base Config Loader**
- Load from torchtune recipe (via `tune cp`)
- Load from custom file path
- Return as Python dict

**3. Merge Engine**
- Deep merge for nested dicts
- Replace for scalars and lists
- See Appendix A for full semantics

**4. Cartesian Product Generator**
- Generate all combinations of variable values
- Create (variables, run_id) pairs

**5. Config Generator**
- For each variable combination:
  - Start with base config
  - Merge controls
  - Merge variable values
  - Write to configs/run_NNN.yaml

**6. Mapping File Writer**
- Create run_mapping.yaml showing which parameters each run uses

---

## Phase 1 Scope

### Included

✅ **experiment.yaml format**
- Variables and controls sections
- Load from torchtune recipe OR custom config file
- Nested dict overrides
- Full factorial sweep (cartesian product)

✅ **Validation**
- Pydantic schema validation
- Required field checking
- Type checking (lists contain valid values)
- Base config exists
- Optional: `tune validate` on generated configs

✅ **CLI**
- `python -m torchtune_config_writer generate experiment.yaml`
- Clear error messages

✅ **Output**
- One torchtune YAML per variable combination
- run_mapping.yaml for traceability

✅ **Testing**
- Unit tests for merge logic
- Integration tests with real torchtune configs
- Validation tests

### Deferred to Phase 2+

❌ Python fluent/builder API (can add if needed)
❌ Variable substitution in paths (`{lora_rank}`)
❌ Metadata tracking (.meta.yaml files)
❌ Multiple loading methods
❌ Advanced sweep strategies (random, conditional)
❌ Evaluation config generation
❌ SLURM script generation

---

## Usage

### Basic Workflow

**1. Create experiment.yaml**
```yaml
experiment_name: my_experiment
base_config: llama3_2/1B_lora_single_device

variables:
  lora_rank: [8, 16, 32]

controls:
  learning_rate: 3e-4
  dataset:
    data_files: /path/to/data.json
```

**2. Generate configs**
```bash
python -m torchtune_config_writer generate experiment.yaml
```

Output:
```
✓ Loaded base config: llama3_2/1B_lora_single_device
✓ Generated 3 configs in configs/
  - configs/run_000.yaml (lora_rank=8)
  - configs/run_001.yaml (lora_rank=16)
  - configs/run_002.yaml (lora_rank=32)
✓ Created run_mapping.yaml
```

**3. (Optional) Validate**
```bash
for f in configs/run_*.yaml; do
  tune validate $f
done
```

**4. Submit to SLURM**
```bash
sbatch --array=0-2 run_experiment.sh
```

### With Custom Base Config

If you've adapted a config for your GPU:

```yaml
experiment_name: my_experiment
base_config_file: /home/user/configs/my_v100_optimized.yaml

variables:
  learning_rate: [1e-4, 3e-4, 5e-4]

controls:
  epochs: 3
```

Tool loads your custom config, applies overrides.

---

## Implementation Plan

### Phase 1 Tasks

**1. Schema Definition** (schema.py)
- [ ] Pydantic models for experiment.yaml
- [ ] Validation rules
- [ ] Type hints
- [ ] Tests

**2. Base Config Loading** (loaders.py)
- [ ] Load from torchtune recipe (via `tune cp`)
- [ ] Load from file path
- [ ] Error handling
- [ ] Tests

**3. Merge Engine** (merge.py)
- [ ] Deep merge implementation
- [ ] Scalar/list replace logic
- [ ] Edge case handling
- [ ] Tests (see Appendix A for test cases)

**4. Cartesian Product Generator** (sweep.py)
- [ ] Generate all variable combinations
- [ ] Assign run IDs
- [ ] Create parameter mappings
- [ ] Tests

**5. Config Generator** (generator.py)
- [ ] Orchestrate: load → merge → write
- [ ] Apply controls
- [ ] Apply variable values
- [ ] Write YAML files
- [ ] Create run_mapping.yaml
- [ ] Tests

**6. CLI** (__main__.py)
- [ ] Argument parsing
- [ ] `generate` command
- [ ] Error messages
- [ ] Progress output
- [ ] Tests

**7. Integration Tests** (tests/integration/)
- [ ] End-to-end: experiment.yaml → configs
- [ ] Real torchtune base configs
- [ ] Validation with `tune validate`

**8. Documentation**
- [ ] Update README with Phase 1 usage
- [ ] Example experiment.yaml files
- [ ] Troubleshooting guide

---

## File Structure

```
torchtune_config_writer/
├── torchtune_config_writer/
│   ├── __init__.py
│   ├── schema.py          # Pydantic models
│   ├── loaders.py         # Base config loading
│   ├── merge.py           # Dict merge logic
│   ├── sweep.py           # Cartesian product generation
│   ├── generator.py       # Main config generation orchestrator
│   └── __main__.py        # CLI entry point
├── tests/
│   ├── test_schema.py
│   ├── test_loaders.py
│   ├── test_merge.py
│   ├── test_sweep.py
│   ├── test_generator.py
│   └── integration/
│       └── test_end_to_end.py
├── appendices/
│   ├── A_merge_semantics.md
│   └── F_experiment_definition.md
├── example_configs/
│   └── (example experiment.yaml files)
├── README.md
├── SPEC.md (this file)
├── SETUP.md
├── CLAUDE.md
├── DEFINITIONS.md
└── pyproject.toml
```

---

## Appendices

- **[Appendix A: Merge Semantics](appendices/A_merge_semantics.md)** - Detailed merge rules and edge cases
- **[Appendix F: Experiment Definition Format](appendices/F_experiment_definition.md)** - Complete experiment.yaml specification

---

## Success Criteria

Phase 1 is successful when:

1. ✅ User can write experiment.yaml defining variables and controls
2. ✅ Tool generates correct torchtune configs for all combinations
3. ✅ Generated configs validate with `tune validate`
4. ✅ Merge semantics work correctly (nested dicts, scalars, lists)
5. ✅ Clear error messages for invalid inputs
6. ✅ Test coverage for critical paths
7. ✅ Documentation is clear and complete
8. ✅ Can be used by another researcher without hand-holding

---

## Future Enhancements (Phase 2+)

After Phase 1 is solid, consider:
- Python API for programmatic sweep generation
- Variable substitution in paths
- Metadata tracking for provenance
- SLURM script generation
- Evaluation config generation
- Smart defaults from prior runs
- Science vs. engineering parameter separation
