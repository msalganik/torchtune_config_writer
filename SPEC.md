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

1. [System Context](#system-context)
2. [Project Goal](#project-goal)
3. [Core Concept](#core-concept)
4. [Design Decisions](#design-decisions)
5. [Architecture](#architecture)
6. [Phase 1 Scope](#phase-1-scope)
7. [Usage](#usage)
8. [Implementation Plan](#implementation-plan)
9. [File Structure](#file-structure)
10. [Appendices](#appendices)

---

## System Context

### Part of cruijff_kit v2

This component is being designed for integration into **cruijff_kit v2**, a research experiment orchestration system for LLM fine-tuning and evaluation.

**cruijff_kit v2 Components:**
- **torchtune config generation** (this component) - ✅ Being spec'd
- **Inspect AI evaluation** - 🚧 Future
- **DSPy optimization** - 🚧 Future
- **SLURM orchestration** - 🚧 Future

### Future Architecture Decisions (Non-Blocking for Phase 1)

**NOTE: These are future considerations for cruijff_kit v2 integration.
They do NOT block Phase 1 implementation, which is fully standalone.**

The following architectural decisions for cruijff_kit v2 integration are still being determined:

- [ ] **Monorepo structure**: Final organization of components
- [ ] **Command interface**: `cruijff-kit torchtune generate` vs other patterns
- [ ] **Component communication**: File-based, API-based, or hybrid
- [ ] **experiment.yaml schema**: Unified format across all frameworks

### Phase 1 Approach: Standalone-Capable, Integration-Ready

This component is being spec'd to be:

**Standalone-capable:**
- Can be used independently: `cruijff-kit torchtune generate experiment.yaml`
- Has both CLI and Python API
- Works without other cruijff_kit components
- Useful for development and testing

**Integration-ready:**
- Will be integrated as `cruijff_kit.torchtune` in monorepo
- Can be called by cruijff_kit orchestrator
- Can be called by Claude Code skills
- Designed for loose coupling with other components

### Usage Modes

**Mode 1: Manual (Human researchers)**
```bash
# Researcher writes experiment.yaml manually
vim experiment.yaml

# Researcher runs generation
cruijff-kit torchtune generate experiment.yaml

# Researcher submits to SLURM manually
sbatch submit_script.sh
```

**Mode 2: Skills (Claude Code automation)**
```python
# Skills can use CLI
subprocess.run(["cruijff-kit", "torchtune", "generate", "experiment.yaml"])

# Or import API
from cruijff_kit.torchtune import generate_configs
result = generate_configs("experiment.yaml")
```

**Mode 3: Integrated (cruijff_kit orchestrator)**
```python
# Called by main orchestrator
from cruijff_kit.torchtune import generate_configs
result = generate_configs(experiment_dict)
# Orchestrator uses result to coordinate with other components
```

### Integration Points

**This component provides:**
- Input: `experiment.yaml` (or dict)
- Output: Generated torchtune config files in `configs/` directory
- Output: `run_mapping.yaml` mapping run IDs to parameters
- API: `generate_configs()` function returning structured results

**This component depends on:**
- `torchtune` CLI (for `tune cp` to load base configs)
- Optional: `torchtune` CLI (for `tune validate` to validate generated configs)

**This component will be used by:**
- SLURM orchestration component (reads `run_mapping.yaml` to submit jobs)
- Evaluation component (reads `run_mapping.yaml` to map checkpoints to parameters)
- Claude Code skills (calls CLI or API)

### See Also

- [CRUIJFF_KIT_V1_SUMMARY.md](CRUIJFF_KIT_V1_SUMMARY.md) - Analysis of v1 architecture
- [REFACTORING_PLAN.md](REFACTORING_PLAN.md) - Migration rationale from v1 to v2

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
# Option 1: Use torchtune recipe (run 'tune ls' to see available configs)
base_config: "llama3_2/1B_lora_single_device"

# Option 2: Use custom file
base_config_file: "/path/to/my_custom_config.yaml"

# Note: Only ONE of base_config or base_config_file can be specified.
# Specifying both will raise a ValidationError.
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

**Decision**: Two-stage validation using minimal pre-checks and torchtune's validator

**Stage 1 - Pre-generation (our tool):**
- Validate experiment.yaml against schema
- Verify base_config exists (via `tune cp` test)
- Check variables have non-empty value lists
- Ensure no duplicate variable names

**Stage 2 - Post-generation (tune validate):**
- Run `tune validate` on each generated config
- Leverage torchtune's comprehensive validation:
  - YAML syntax validity
  - Component existence (`_component_` paths resolve)
  - Required parameters present
  - No unexpected parameters
  - Type compatibility via Python signature binding

**Rationale**:
- Don't reinvent the wheel - `tune validate` is authoritative
- **Critical for SLURM**: Find config errors before queue submission
- Our minimal pre-checks prevent wasted generation time

**Implementation**: After generating all configs, run `tune validate` on each. Continue validating all configs even if one fails, then report summary.

---

### 5. Error Handling Strategy

**Decision**: Clear error hierarchy with actionable messages

**Error Classes:**
```python
ConfigWriterError (base)
├── ValidationError
│   ├── BaseConfigNotFoundError    # base_config doesn't exist
│   ├── ExperimentSchemaError      # experiment.yaml invalid
│   └── EmptyVariableError         # variable has no values
├── MergeError
│   └── TypeMismatchError          # incompatible types in merge
└── TorchtuneValidationError       # tune validate failed
```

**Philosophy:**
- Fail fast with clear, actionable error messages
- Include context (which variable, which line, suggested fix)
- Example:
  ```
  BaseConfigNotFoundError: Config 'llama3/invalid' not found.
    Available configs starting with 'llama3':
      - llama3/8B_full_single_device
      - llama3/8B_lora_single_device
    Run 'tune ls' for full list.
  ```

---

### 6. No Builder API in Phase 1

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
- Framework must be "torchtune" (errors on other values in Phase 1)
- Type checking (lists contain valid values)
- Base config exists (via `tune cp` test)
- Post-generation: `tune validate` on all generated configs

✅ **CLI**
- `python -m torchtune_config_writer generate experiment.yaml`
- Clear error messages

✅ **Output**
- One torchtune YAML per variable combination
- run_mapping.yaml for traceability
- Organized folder structure for multiple experiments

✅ **Testing**
- Unit tests for merge logic
- Integration tests with real torchtune configs
- Validation tests

### Deferred to Phase 2+

❌ Python API (programmatic access)
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

## Output Folder Structure

**Decision**: Experiment name + timestamp pattern for unique, organized outputs

**Structure:**
```
outputs/
├── lora_rank_sweep_20250112_143022/
│   ├── experiment.yaml          # Copy of source for reproducibility
│   ├── configs/
│   │   ├── run_000.yaml         # Generated torchtune configs
│   │   ├── run_001.yaml
│   │   ├── run_002.yaml
│   │   └── run_003.yaml
│   ├── run_mapping.yaml         # Maps run_XXX to variable values
│   └── validation_report.txt    # Results of tune validate
```

**Key Features:**
- **Unique folders**: Timestamp prevents collisions
- **Self-contained**: Everything for one experiment in one folder
- **Reproducible**: Original experiment.yaml preserved
- **Traceable**: Clear naming shows what and when

**Output Path Resolution:**
1. Command line `--output-dir` (highest priority)
2. experiment.yaml `output.experiment_dir` field
3. Auto-generated: `outputs/{experiment_name}_{timestamp}/`

**Example:**
```bash
# Auto-generated unique folder
$ python torchtune_config_writer.py experiment.yaml
Created: outputs/lora_rank_sweep_20250112_143022/

# User-specified folder
$ python torchtune_config_writer.py experiment.yaml --output-dir my_results/exp1
Created: my_results/exp1/
```

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
