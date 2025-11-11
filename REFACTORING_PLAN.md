# Torchtune Config Writer - Refactoring Plan

**Date**: 2025-11-11
**Purpose**: Refactor torchtune config generation from markdown-driven to structured, testable system
**Status**: Proposal for review

---

## Executive Summary

We're refactoring the config generation component of cruijff_kit to replace the current markdown → Claude skill → template system with a more structured, testable approach using `experiment.yaml`.

**Current system:** experiment_summary.md (markdown) → Claude parses → setup_finetune.yaml → setup_finetune.py fills templates → finetune.yaml + finetune.slurm

**Proposed system:** experiment.yaml (structured) → Python tool → finetune.yaml (+ optional SLURM script)

**Benefits:**
- ✅ Structured, machine-readable experiment definitions
- ✅ Testable (not dependent on Claude parsing markdown)
- ✅ Supports both conversational (Claude creates experiment.yaml) and manual workflows
- ✅ Variables vs. controls mirrors scientific thinking
- ✅ Can start from torchtune recipes OR custom configs
- ✅ Type-safe validation with Pydantic

**Phase 1 Scope:** Fine-tuning config generation only. Evaluation, SLURM scripts, and other features deferred.

---

## Table of Contents

1. [Current System Analysis](#current-system-analysis)
2. [Problems with Current System](#problems-with-current-system)
3. [Proposed Solution](#proposed-solution)
4. [Key Design Decisions](#key-design-decisions)
5. [What We're Learning from cruijff_kit](#what-were-learning-from-cruijff_kit)
6. [Implementation Architecture](#implementation-architecture)
7. [Example Workflows](#example-workflows)
8. [Migration Path](#migration-path)
9. [Success Criteria](#success-criteria)
10. [Timeline & Milestones](#timeline--milestones)

---

## Current System Analysis

### How It Works Now

**Step 1: Design experiment** (design-experiment skill)
- User + Claude have conversation about experiment
- Claude creates `experiment_summary.md` with:
  - Run table (model, batch_size, lora_rank, etc.)
  - Dataset paths
  - Resource estimates
  - Evaluation plan

**Step 2: Scaffold configs** (scaffold-torchtune skill)
- Claude parses experiment_summary.md
- For each run, generates `setup_finetune.yaml` (simplified config)
- Contains: hyperparameters, paths, SLURM settings

**Step 3: Generate final configs** (setup_finetune.py)
- Reads `setup_finetune.yaml`
- Loads `finetune_template.yaml` (torchtune config with placeholders)
- Applies special logic:
  - Dataset format detection (.json vs .parquet)
  - LoRA alpha calculation (2 × rank)
  - Path construction
  - Multi-GPU handling
- Generates:
  - `finetune.yaml` (complete torchtune config)
  - `finetune.slurm` (SLURM submission script)

### What Works Well

1. **Two-tier config system** - setup_finetune.yaml is simple, finetune.yaml is complete
2. **Special handling** - Knows about torchtune quirks (dataset formats, component paths)
3. **SLURM integration** - Generates submission scripts
4. **Multi-GPU support** - Switches recipes and resource allocations
5. **Flexible** - Can override via CLI or config file

### Current File Structure

```
cap_viz_test_2025-10-28/
├── experiment_summary.md          # Claude-generated
├── 1B_rank8/
│   ├── setup_finetune.yaml        # Input to setup_finetune.py
│   ├── finetune.yaml             # Generated torchtune config
│   └── finetune.slurm            # Generated SLURM script
└── 3B_rank8/
    ├── setup_finetune.yaml
    ├── finetune.yaml
    └── finetune.slurm
```

---

## Problems with Current System

### 1. Brittle Markdown Parsing

**Problem:** experiment_summary.md is markdown intended for humans. Claude must parse tables, extract parameters, infer structure.

**Risk:** Changes to markdown format break parsing. Hard to test. Error-prone.

**Example:**
```markdown
| Run Name | Model | LoRA Rank | Batch Size |
|----------|-------|-----------|------------|
| 1B_rank8 | Llama-3.2-1B-Instruct | 8 | 16 |
```

Claude must:
- Parse table
- Extract values
- Map column names to parameters
- Handle variations in formatting

### 2. Hard to Test

**Problem:** Config generation logic is spread across:
- Claude's markdown parsing (in skill)
- setup_finetune.py template logic
- Coordination between them

**Result:** Can't write automated tests. Must test by running full workflow.

### 3. Not User-Editable

**Problem:** Advanced users can't directly edit experiment definition without Claude. Must go through conversation.

**Workaround:** Manually edit experiment_summary.md, but then Claude can't re-generate from it.

### 4. Template Maintenance

**Problem:** finetune_template.yaml must stay in sync with torchtune's config format. As torchtune evolves, template needs updates.

**Alternative:** Start from torchtune's official recipe configs, apply overrides.

---

## Proposed Solution

### Core Concept

Replace markdown with structured `experiment.yaml`:

```yaml
experiment_name: lora_rank_study
base_config: llama3_2/1B_lora_single_device  # Or: base_config_file: /path/to/custom.yaml

variables:  # What varies (independent variables)
  lora_rank: [8, 16, 32, 64]

controls:   # What's constant (controlled variables)
  learning_rate: 3e-4
  batch_size: 16
  epochs: 2
  dataset:
    data_files: /path/to/data.json
    packed: true
```

**Tool generates:**
- 4 complete torchtune configs (one per lora_rank value)
- run_mapping.yaml (maps run IDs to parameters)

### Two Workflows

**A) Conversational (primary)**
1. User + Claude discuss experiment
2. Claude creates experiment.yaml (instead of experiment_summary.md)
3. User runs: `python -m torchtune_config_writer generate experiment.yaml`
4. Gets configs

**B) Manual (advanced users)**
1. User writes experiment.yaml in editor
2. Runs generation tool
3. Gets configs

Both produce same result. experiment.yaml is the source of truth.

---

## Key Design Decisions

### 1. Base Config Loading

**Decision:** Support both torchtune recipes AND custom config files

```yaml
# Option A: Use torchtune's official recipe
base_config: "llama3_2/1B_lora_single_device"

# Option B: Use your custom config (e.g., GPU-adapted)
base_config_file: "/scratch/configs/my_v100_optimized.yaml"
```

**Rationale:**
- Torchtune maintains 100+ proven recipes
- Users can adapt one for their GPU, then generate sweeps from it
- Flexibility for both beginners and experts

**Implementation:** Use `tune cp` for recipes, read file for custom.

### 2. Variables vs Controls

**Decision:** Explicit separation in experiment.yaml

**Why:**
- Mirrors scientific experimental design
- Makes experimental design obvious
- Easy to review ("What am I actually testing?")
- Good documentation for papers/repos

**Example:**
```yaml
variables:  # What I'm testing
  lora_rank: [8, 16, 32]
  learning_rate: [1e-4, 3e-4]

controls:   # What's held constant
  epochs: 3
  batch_size: 4
```

Generates 3 × 2 = 6 configs (full factorial).

### 3. Deep Merge Strategy

**Decision:** Start from base config, recursively merge overrides

**Rationale:**
- Can override specific nested fields without rewriting entire sections
- Predictable behavior (scalars replace, dicts merge, lists replace)
- Well-defined semantics (see Appendix A)

**Example:**
```yaml
controls:
  dataset:
    data_files: /new/path.json  # Only changes this field
    # Other dataset fields from base config preserved
```

### 4. No Templates

**Decision:** Don't use templates. Start from actual torchtune configs and merge.

**Rationale:**
- Templates require maintenance as torchtune evolves
- Torchtune's configs are the authoritative source
- Merge approach is more flexible and future-proof

### 5. Phase 1: Config Generation Only

**In scope:**
- experiment.yaml → finetune.yaml generation
- Variables/controls with cartesian product
- Pydantic validation
- Support torchtune recipes and custom configs
- run_mapping.yaml for traceability

**Out of scope (Phase 2+):**
- SLURM script generation (keep using setup_finetune.py for now)
- Evaluation config generation
- Metadata tracking
- Python fluent API
- Science vs. engineering parameter separation

---

## What We're Learning from cruijff_kit

### Important Features to Preserve

**1. Dataset Format Handling**

cruijff_kit handles .json vs .parquet automatically:

```python
if args.dataset_ext == '.json':
    config["dataset"]["source"] = "json"
    config["dataset"]["data_files"] = ...
elif args.dataset_ext == '.parquet':
    config["dataset"]["data_dir"] = ...
```

**Our approach:** Let users specify full dataset config in controls. They know their format.

```yaml
controls:
  dataset:
    _component_: torchtune.datasets.instruct_dataset
    source: json
    data_files: /path/to/data.json
    field: train
```

**2. Parameter Derivation**

cruijff_kit auto-calculates lora_alpha = 2 × lora_rank.

**Our approach Phase 1:** User specifies both in controls:
```yaml
controls:
  lora_rank: 8
  lora_alpha: 16
```

**Phase 2+:** Could add smart defaults or warnings.

**3. Multi-GPU Support**

cruijff_kit switches from `lora_finetune_single_device` to `lora_finetune_distributed` when gpus > 1.

**Our approach Phase 1:** Out of scope. User specifies correct recipe.

**Phase 2+:** Could add SLURM generation with multi-GPU logic.

**4. SLURM Integration**

cruijff_kit generates both finetune.yaml and finetune.slurm.

**Our approach Phase 1:** Only generate finetune.yaml.

**Interim solution:** Users can still use setup_finetune.py for SLURM script generation, or write their own job array scripts.

### Code Patterns to Adopt

**1. Config File + CLI Override Pattern**

setup_finetune.py loads YAML, allows CLI overrides:
```python
config_data = yaml.safe_load(config_file)
# Merge with argparse args, CLI takes precedence
```

**We'll use:** Pydantic models load from experiment.yaml, validate structure.

**2. SLURM-Only Parameters**

setup_finetune.py has `SLURM_ONLY` list - parameters that don't go in torchtune config.

**We'll use:** All SLURM stuff out of scope for Phase 1.

**3. Validation Lists**

setup_finetune.py validates against allowed values:
```python
VALID_LR_SCHEDULERS = [
    'get_cosine_schedule_with_warmup',
    'get_linear_schedule_with_warmup',
    ...
]
```

**We'll use:** Pydantic validation for schema. Torchtune's `tune validate` for config correctness.

---

## Implementation Architecture

### Components

```
experiment.yaml
      ↓
[1. Schema (Pydantic)]  ← Validation
      ↓
[2. Base Config Loader]
      ├─ tune cp llama3_2/1B...  (torchtune recipe)
      └─ read file               (custom config)
      ↓
  Base Config Dict
      ↓
[3. Sweep Generator]
      ↓
  Variable Combinations
  [(rank=8), (rank=16), (rank=32), ...]
      ↓
[4. Merge Engine]  ← For each combination:
      ├─ Start with base config
      ├─ Deep merge controls
      └─ Deep merge variable values
      ↓
  N Config Dicts
      ↓
[5. YAML Writer]
      ↓
configs/
  ├─ run_000.yaml
  ├─ run_001.yaml
  ├─ run_002.yaml
  └─ run_mapping.yaml
```

### File Structure

```
torchtune_config_writer/
├── torchtune_config_writer/
│   ├── __init__.py
│   ├── schema.py          # Pydantic models
│   ├── loaders.py         # Load base configs
│   ├── merge.py           # Deep merge logic
│   ├── sweep.py           # Cartesian product
│   ├── generator.py       # Orchestrator
│   └── __main__.py        # CLI
├── tests/
│   ├── test_schema.py
│   ├── test_loaders.py
│   ├── test_merge.py
│   ├── test_sweep.py
│   ├── test_generator.py
│   └── integration/
│       └── test_end_to_end.py
├── docs/
│   ├── README.md
│   ├── SPEC.md
│   ├── REFACTORING_PLAN.md (this file)
│   └── appendices/
│       ├── A_merge_semantics.md
│       └── F_experiment_definition.md
└── pyproject.toml
```

### Testing Strategy

**Unit Tests:**
- Schema validation (Pydantic)
- Merge logic (edge cases from Appendix A)
- Cartesian product generation
- Base config loading

**Integration Tests:**
- End-to-end: experiment.yaml → configs
- Real torchtune recipes as base
- Validate generated configs with `tune validate`

**Comparison Tests:**
- Generate same experiment with old and new systems
- Compare resulting finetune.yaml files
- Ensure compatibility

---

## Example Workflows

### Example 1: Simple LoRA Rank Sweep

**experiment.yaml:**
```yaml
experiment_name: lora_rank_study
base_config: llama3_2/1B_lora_single_device

variables:
  lora_rank: [8, 16, 32, 64]

controls:
  learning_rate: 3e-4
  lora_alpha: 16
  batch_size: 4
  epochs: 3
  dataset:
    _component_: torchtune.datasets.instruct_dataset
    source: json
    data_files: /scratch/data/words_8L_80P_5000.json
    field: train
    packed: true
  output_dir: /scratch/results/lora_study
  metric_logger:
    _component_: torchtune.training.metric_logging.WandBLogger
    project: capitalization
    mode: offline
```

**Generation:**
```bash
python -m torchtune_config_writer generate experiment.yaml
```

**Output:**
```
✓ Loaded base config: llama3_2/1B_lora_single_device
✓ Generated 4 configs in configs/
  - configs/run_000.yaml (lora_rank=8)
  - configs/run_001.yaml (lora_rank=16)
  - configs/run_002.yaml (lora_rank=32)
  - configs/run_003.yaml (lora_rank=64)
✓ Created run_mapping.yaml
```

### Example 2: Two-Variable Grid with Custom Base

**experiment.yaml:**
```yaml
experiment_name: lr_batch_sweep
base_config_file: /scratch/configs/v100_optimized.yaml  # Pre-adapted for V100 GPUs

variables:
  learning_rate: [1e-4, 3e-4, 5e-4]
  batch_size: [2, 4, 8]

controls:
  epochs: 5
  seed: 42
```

Generates 3 × 3 = 9 configs.

### Example 3: Integration with Claude Design-Experiment Skill

**Modified design-experiment skill:**
- Still has conversational design process
- Still asks all the same questions
- Still verifies resources
- **NEW:** Outputs experiment.yaml instead of experiment_summary.md

**Modified scaffold-experiment skill:**
- Runs: `python -m torchtune_config_writer generate experiment.yaml`
- For each generated config, creates SLURM script (or uses existing setup_finetune.py)

**User experience:** Nearly identical, but more robust backend.

---

## Migration Path

### Phase 1: Parallel Systems

**Keep:**
- Existing design-experiment skill (creates experiment_summary.md)
- Existing scaffold-torchtune skill (creates setup_finetune.yaml)
- Existing setup_finetune.py

**Add:**
- New torchtune_config_writer package
- Can be used manually or via new skills

**Benefits:**
- No breaking changes
- Can test new system alongside old
- Gradual migration

### Phase 2: Update Design-Experiment Skill

**Modify design-experiment skill to:**
- Output experiment.yaml instead of experiment_summary.md
- Still conversational, same questions
- More structured output

**Create new scaffold-experiment-v2 skill:**
- Uses torchtune_config_writer
- Calls: `python -m torchtune_config_writer generate`
- Still creates SLURM scripts (using setup_finetune.py or custom logic)

### Phase 3: Deprecate Old System

**After validation:**
- Mark old skills as deprecated
- Update documentation to use new workflow
- Keep old code for reference

---

## Success Criteria

Phase 1 is successful when:

1. ✅ **Correctness**: Generated configs are valid and work for fine-tuning
2. ✅ **Validation**: All generated configs pass `tune validate`
3. ✅ **Testability**: 80%+ test coverage on critical paths (merge, validation, generation)
4. ✅ **Usability**: Another researcher can use it without help
5. ✅ **Documentation**: Clear examples and troubleshooting guide
6. ✅ **Performance**: Generates 100 configs in < 10 seconds
7. ✅ **Flexibility**: Supports both torchtune recipes and custom configs
8. ✅ **Integration**: Can be used standalone or with modified design-experiment skill

---

## Open Questions for Discussion

1. **SLURM script generation**: Should Phase 1 include this? Or keep using setup_finetune.py?

2. **Parameter defaults**: Should we have smart defaults (like lora_alpha = 2 × rank) or require explicit specification?

3. **Error handling**: How verbose should validation errors be? Show all errors or fail fast?

4. **Dataset handling**: Require users to specify full dataset config, or provide helpers for common cases (.json, .parquet)?

---

## Next Steps

1. **Review this plan** with team
2. **Discuss open questions** and make decisions
3. **Validate approach** with small prototype
4. **Begin implementation** (Week 1-2 tasks)
5. **Iterate based on feedback**

---

## References

- [SPEC.md](SPEC.md) - Technical specification
- [Appendix A: Merge Semantics](appendices/A_merge_semantics.md) - Detailed merge rules
- [Appendix F: Experiment Definition Format](appendices/F_experiment_definition.md) - experiment.yaml specification
- [DEFINITIONS.md](DEFINITIONS.md) - Terminology reference
- [cruijff_kit repository](https://github.com/niznik-dev/cruijff_kit) - Current system
