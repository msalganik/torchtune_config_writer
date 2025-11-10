# Appendix F: Experiment Definition Format

**Purpose**: This appendix defines the `experiment.yaml` format for declarative experiment definitions. This is the "connector between scientist's brain and software" - translating how scientists think about experiments into executable config sweeps.

**Audience**: Scientists defining experiments, implementers of the config generation tool.

**Related sections**:
- See Appendix C for high-level builder methods
- See main SPEC.md for builder API and architecture
- See Phase 4 of SPEC.md for implementation timeline

**Status**: 🚧 Simplified for Phase 1 - Basic features only

**Phase 1 Scope** (Simplified):
- ✅ Basic experiment.yaml schema
- ✅ Simple variable sweeps (lists only)
- ✅ Fixed controls
- ✅ Direct mapping to builder API
- ✅ Basic validation (schema + existence checks)
- ✅ Simple `generate` command

**Phase 2+ Features** (Deferred):
- ❌ Variable reference syntax `{variable_name}` in paths
- ❌ Auto-generated naming templates
- ❌ manifest.json and experiment_summary.md generation
- ❌ Multi-level validation (Levels 1-4)
- ❌ Fuzzy matching for suggestions
- ❌ Claude skill integration
- ❌ Advanced sweep patterns (random search, conditional params)

---

## Table of Contents

1. [Overview](#overview)
2. [Design Decisions](#design-decisions)
3. [Format Specification](#format-specification)
4. [Complete Example](#complete-example)
5. [Evaluation Configuration](#evaluation-configuration)
6. [Mapping to Builder API](#mapping-to-builder-api)
7. [Validation Rules](#validation-rules)
8. [Error Handling](#error-handling)
9. [Tool Implementation](#tool-implementation)
10. [Future Extensions](#future-extensions)

---

## Overview

### The Problem

Scientists think about experiments in terms of:
- Research questions and hypotheses
- Independent variables to sweep
- Controlled variables to keep constant
- Expected outcomes

But implementing this requires:
- Writing Python loops
- Managing file paths
- Generating configs
- Tracking what changed

### The Solution

**experiment.yaml**: A declarative format that captures scientific thinking directly.

```yaml
experiment:
  name: "lora_rank_study"
  question: "How does LoRA rank affect performance?"

variables:
  lora_rank: [8, 16, 32, 64]

controls:
  learning_rate: 3e-4
  dataset: "data/alpaca.json"
```

Tool generates → N configs → Submit to SLURM → Get results

---

## Design Decisions

### Core Principles

1. **Experiment-centric, not code-centric**: Match scientist's mental model
2. **Self-documenting**: Experiment files should be readable 6 months later
3. **Framework-agnostic design**: Support torchtune (Phase 1), others later (Phase 2+)
4. **Separation of concerns**: Science ≠ Infrastructure (no SLURM details here)
5. **Standalone files**: No inheritance/references between experiments
6. **Simple for common cases**: Single and two-variable sweeps should be trivial

---

### Key Decisions (Phase 1 Simplified)

**Decision 1: Primary Use Cases (Phase 1)**
- Single-variable sweep
- Two-variable factorial grid
- Defer: 3+ variables, random search, conditional parameters

**Decision 2: No Inheritance**
- Each experiment file is standalone and self-contained
- No `based_on` or references to other experiments
- Copy-paste is acceptable; clarity beats DRY

**Decision 3: User Writes experiment.yaml Manually**
- Phase 1: User writes experiment.yaml in editor
- Phase 2+: Add Claude skill for interactive design
- Simple enough to write by hand

**Decision 4: Separation from SLURM**
- experiment.yaml defines SCIENCE (what to test)
- Separate tool handles INFRASTRUCTURE (SLURM submission)
- Different clusters, same experiment definition

**Decision 5: Variables vs Controls**
```yaml
variables:   # What sweeps (independent variables)
  param1: [val1, val2, val3]

controls:    # What's constant (controlled variables)
  param2: fixed_value
```

**Decision 6: Simple Lists Only (Phase 1)**
```yaml
variables:
  lora_rank: [8, 16, 32]  # Just explicit lists
  # No ranges, no distributions, no sampling
```

**Decision 7: Fixed Paths Only (Phase 1)**
```yaml
controls:
  dataset: "data/alpaca.json"  # Fixed path, no variable substitution
  # Phase 2+: Add {variable_name} syntax
```

**Decision 8: User-Specified Naming (Phase 1)**
```yaml
output:
  configs_dir: "configs/"
  results_dir: "results/"
  # User creates subdirs manually or in Python wrapper
  # Phase 2+: Add naming templates
```

**Decision 9: Tight Mapping to Builder API**
- YAML keys map to builder methods: `learning_rate` → `with_learning_rate()`
- Fallback to `override()` for parameters without high-level methods
- Special handling for multi-parameter methods (e.g., `lora_rank` + `lora_alpha`)

**Decision 10: Multi-Parameter Methods**
- `lora_rank` + `lora_alpha` must both be specified (error if only one)
- Can be in different sections (rank in variables, alpha in controls)
- Both in variables → cartesian product

**Decision 11: Basic Validation Only (Phase 1)**
- Validate YAML schema (required fields, types)
- Check base config exists
- Check dataset file exists
- Warn if > 100 configs
- Phase 2+: Fuzzy matching, multi-level validation

**Decision 12: Minimal CLI (Phase 1)**
- Just `generate` command
- Phase 2+: Add `validate`, `list` commands

---

## Format Specification

### Top-Level Structure

```yaml
experiment:
  # Experiment metadata

framework_config:
  # Framework-specific configuration

variables:
  # Parameters to sweep (generate grid)

controls:
  # Parameters held constant

output:
  # Output directory organization
```

---

### Section 1: experiment (Metadata)

**Purpose**: Scientific documentation and organization

```yaml
experiment:
  name: "lora_sample_size_study"      # REQUIRED: Unique identifier
  framework: "torchtune"               # REQUIRED: "torchtune" in Phase 1

  # Optional but encouraged
  question: "How does performance vary with LoRA rank and sample size?"
  hypothesis: "Larger ranks need more samples to show benefit"
  researcher: "Alice"
  date: "2025-01-09"  # When experiment was DESIGNED (not run)
  tags: ["lora", "sample-efficiency", "llama3.1"]
  notes: |
    Following up on initial results that showed rank=16 performed
    well. Testing hypothesis that higher ranks need more data.
```

**Field Specifications**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ Yes | Unique experiment identifier. Used in output paths. |
| `framework` | string | ✅ Yes | Framework to use. Phase 1: only "torchtune" supported. |
| `question` | string | ❌ No | Research question this experiment addresses. |
| `hypothesis` | string | ❌ No | Expected outcome or prediction. |
| `researcher` | string | ❌ No | Who designed/ran this experiment. |
| `date` | string | ❌ No | When experiment was designed (YYYY-MM-DD). NOT when jobs ran. |
| `tags` | list[string] | ❌ No | Tags for search/organization (e.g., ["lora", "efficiency"]). |
| `notes` | string | ❌ No | Free-form observations, context, or rationale. |

**Notes**:
- `date` is when experiment was **designed**, not when SLURM jobs run (that's in job logs)
- Only `name` and `framework` are required; rest is optional but encouraged
- Tags help Claude search: "Show me all 'lora' experiments"

---

### Section 2: framework_config (Torchtune Settings)

**Purpose**: Framework-specific base configuration

```yaml
framework_config:
  base_config: "llama3_1/8B_lora_single_device"  # REQUIRED
```

**Field Specifications**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `base_config` | string | ✅ Yes | Torchtune base config name (from `tune ls`). |

**Valid Values**:
- Any config shown in `tune ls` output
- Examples: `"llama3_1/8B_lora_single_device"`, `"llama3/8B_full"`, `"mistral/7B_lora_single_device"`

**Notes**:
- This section will expand in Phase 2+ for other frameworks (huggingface, axolotl, etc.)
- For torchtune, only `base_config` is needed; all other params go in `variables`/`controls`

---

### Section 3: variables (Parameter Sweep)

**Purpose**: Define which parameters vary and their values (independent variables)

```yaml
variables:
  lora_rank: [8, 16, 32, 64]
  learning_rate: [1e-5, 3e-5, 5e-5]
```

**Format**: Dictionary of `parameter_name: [list of values]`

**Behavior**:
- Tool generates **full factorial grid** (cartesian product)
- Example above: 4 × 3 = 12 runs
- Each combination becomes one config file

**Supported Parameter Types**:
- Numbers: `learning_rate: [1e-5, 3e-5, 5e-5]`
- Integers: `lora_rank: [8, 16, 32, 64]`
- Strings: `dataset: ["data/alpaca.json", "data/dolly.json"]`
- Booleans: `packed: [true, false]`

**Variable Naming**:
- Must match builder API parameter names (see [Mapping to Builder API](#mapping-to-builder-api))
- Common variables: `learning_rate`, `lora_rank`, `epochs`, `batch_size`, `dataset`, etc.

**Constraints (Phase 1)**:
- Simple lists only (no ranges, distributions, or sampling)
- All values explicit (no `range(8, 64, 8)` syntax)
- Maximum 2-3 variables recommended (avoid explosion: 10×10×10 = 1000 configs!)

---

### Section 4: controls (Fixed Parameters)

**Purpose**: Define parameters held constant across all runs (controlled variables)

```yaml
controls:
  learning_rate: 3e-4
  epochs: 3
  batch_size: 2
  dataset: "data/alpaca.json"
  seed: 42
```

**Format**: Dictionary of `parameter_name: value`

**Behavior**:
- Applied to all generated configs
- Maps to builder API methods (see [Mapping to Builder API](#mapping-to-builder-api))

**Parameter Types**:
- Numbers: `learning_rate: 3e-4`
- Integers: `epochs: 3`
- Strings: `dataset: "data/alpaca.json"`
- Booleans: `packed: true`
- Null: `seed: null`

**Phase 1 Limitation - Fixed Paths Only**:
```yaml
controls:
  dataset: "data/alpaca.json"  # Fixed path
  # Phase 2+: Variable substitution like "data/alpaca_{sample_size}.json"
```

**Workaround for Phase 1**: If you need per-variable datasets, use `_override` with explicit paths for each run (or write Python wrapper).

**Advanced: Nested Overrides**

For complex parameters not covered by high-level methods, use `_override`:

```yaml
controls:
  learning_rate: 3e-4  # Uses with_learning_rate()

  _override:
    optimizer:
      weight_decay: 0.01
      amsgrad: true
    profiler:
      enabled: true
      trace_options:
        profile_memory: true
```

**Override Rules**:
- `_override` lives inside `controls` section (not top-level)
- Applied **after** high-level methods (last wins if conflict)
- Avoid overlapping high-level + override for same parameter (confusing)

---

### Section 5: output (Directory Organization)

**Purpose**: Define where generated configs go

**Phase 1 - Simplified**:
```yaml
output:
  configs_dir: "configs/"       # Where to write generated configs
  results_dir: "results/"       # Where training jobs will write output
  eval_results_dir: "eval_results/"  # Where evaluation outputs go (if evaluation enabled)
```

**Field Specifications**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `configs_dir` | string | ❌ No | Directory for generated configs. Defaults to `./configs/` |
| `results_dir` | string | ❌ No | Directory for training outputs. Defaults to `./results/` |
| `eval_results_dir` | string | ❌ No | Directory for evaluation outputs. Defaults to `./eval_results/` |

**File Organization Convention**:
```
experiment_name/              # Self-contained experiment folder
├── experiment.yaml           # Experiment definition
├── evals/                    # User-written evaluation tasks (if using evaluation)
│   ├── my_eval.py
│   └── data/
├── configs/                  # Generated configs
│   ├── train_000.yaml        # Sequential numbering
│   ├── train_001.yaml
│   ├── eval_000.py           # Generated eval scripts (if evaluation enabled)
│   ├── eval_001.py
│   └── run_mapping.yaml
├── results/                  # Training outputs (jobs write here)
│   ├── run_000/
│   ├── run_001/
│   └── ...
└── eval_results/             # Evaluation outputs (if evaluation enabled)
    ├── run_000/
    ├── run_001/
    └── ...
```

**Naming in Phase 1**:
- Training configs: `train_000.yaml`, `train_001.yaml`, etc.
- Eval scripts: `eval_000.py`, `eval_001.py`, etc.
- Results directories: `results/run_000/`, `results/run_001/`, etc.
- Simple, predictable, no variable substitution needed

**Phase 2+ Enhancement**:
Add `naming_template` field with variable substitution:
```yaml
output:
  configs_dir: "configs/"
  naming_template: "rank{lora_rank}_lr{learning_rate}"
  # Creates: configs/rank8_lr0.0001.yaml, etc.
```

---

### Section 6: evaluation (Optional - Evaluation Configuration)

**Purpose**: Define post-training evaluation using Inspect AI

**Phase 1 - Simplified**:
```yaml
evaluation:
  enabled: true               # Enable evaluation (optional)
  framework: "inspect"        # Phase 1: only "inspect" supported
  checkpoint: "last"          # Phase 1: only "last" checkpoint
  model_format: "hf"          # HuggingFace format (for Inspect AI)

  # Directory containing USER-WRITTEN evaluation tasks
  tasks_dir: "evals/"         # Required if enabled=true

  # List of user-written tasks to run
  tasks:
    - name: "domain_qa"       # Descriptive name
      file: "domain_qa.py"    # Relative to tasks_dir
      function: "domain_qa"   # Task function name (with @task decorator)
      args:                   # Optional: arguments to pass to task
        split: "test"
        max_samples: 100

    - name: "safety_eval"
      file: "safety_eval.py"
      function: "safety_eval"
      # No args - task uses defaults
```

**Field Specifications**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `enabled` | bool | ❌ No | Enable evaluation. Defaults to `false`. |
| `framework` | string | ✅ Yes (if enabled) | Evaluation framework. Phase 1: only `"inspect"` supported. |
| `checkpoint` | string | ❌ No | Which checkpoint to evaluate. Phase 1: only `"last"`. Defaults to `"last"`. |
| `model_format` | string | ❌ No | Model format for evaluation. Defaults to `"hf"` (HuggingFace). |
| `tasks_dir` | string | ✅ Yes (if enabled) | Directory containing user-written evaluation tasks. |
| `tasks` | list | ✅ Yes (if enabled) | List of evaluation tasks to run. |
| `tasks[].name` | string | ✅ Yes | Descriptive name for the task. |
| `tasks[].file` | string | ✅ Yes | Python file containing task (relative to `tasks_dir`). |
| `tasks[].function` | string | ✅ Yes | Function name to import (must have `@task` decorator). |
| `tasks[].args` | dict | ❌ No | Optional arguments to pass to task function. |

**Important Notes**:

1. **User writes evaluation tasks**: The tool does NOT provide built-in evaluation tasks. Users must write their own Inspect AI tasks using the `@task` decorator.

2. **Standard Inspect AI syntax**: Tasks are standard Inspect AI task definitions. See [Inspect AI documentation](https://inspect.aisi.org.uk/) for details.

3. **Model format options**: Two approaches for LoRA fine-tuning:

   **Option A: Full Model Export** (simpler, more storage)
   ```yaml
   controls:
     _override:
       checkpointer:
         _component_: torchtune.training.FullModelHFCheckpointer
         checkpoint_dir: ${output_dir}/checkpoints
         output_dir: ${output_dir}/hf_model
         # save_adapter_weights_only: False (default)
   ```
   - Storage: ~15 GB per checkpoint
   - Evaluation: Direct loading in Inspect AI (`hf/path/to/model`)

   **Option B: Adapter-Only Export** (storage-efficient)
   ```yaml
   controls:
     _override:
       checkpointer:
         _component_: torchtune.training.FullModelHFCheckpointer
         checkpoint_dir: ${output_dir}/checkpoints
         output_dir: ${output_dir}/adapter
         save_adapter_weights_only: True  # Only save adapter weights

   evaluation:
     model_format: "adapter"  # Changed from "hf"
     base_model: "meta-llama/Llama-3.1-8B"  # Base model for adapters
   ```
   - Storage: ~20-160 MB per adapter (200x reduction!)
   - Evaluation: Two sub-options:
     - **Merge adapters** → full model (simple, ~15GB temp storage per eval)
     - **Load adapters directly** (advanced, minimal temp storage, needs Inspect AI compatibility verification)

4. **Separate execution**: Evaluation runs as separate jobs after training completes (not inline during training).

**Phase 2+ Enhancements**:
- Support "best" checkpoint selection (based on validation metrics)
- Support multiple checkpoint evaluation per run
- Support other evaluation frameworks
- Conditional evaluation (only evaluate top N models)

---

## Complete Example

### Example 1: LoRA Rank Study (Phase 1 Simplified)

**Scientific Setup**:
- Research question: How does LoRA rank affect performance?
- Independent variable: lora_rank
- Controlled variables: learning_rate, epochs, dataset

```yaml
# experiment_001.yaml

experiment:
  name: "lora_rank_study"
  framework: "torchtune"

  question: "How does LoRA rank affect fine-tuning performance?"
  researcher: "Alice"
  date: "2025-01-09"
  tags: ["lora", "llama3.1"]

framework_config:
  base_config: "llama3_1/8B_lora_single_device"

variables:
  lora_rank: [8, 16, 32, 64]
  # Single variable: 4 runs

controls:
  learning_rate: 3e-4
  lora_alpha: 32
  epochs: 3
  batch_size: 2
  seed: 42
  dataset: "data/alpaca.json"  # Fixed path (Phase 1)

output:
  configs_dir: "configs/"
  results_dir: "results/"
```

**Generated Outputs**:
```
lora_rank_study/
├── experiment.yaml       # User-written
├── configs/              # Generated by tool
│   ├── train_000.yaml    # lora_rank=8
│   ├── train_001.yaml    # lora_rank=16
│   ├── train_002.yaml    # lora_rank=32
│   ├── train_003.yaml    # lora_rank=64
│   └── run_mapping.yaml
└── results/              # Training outputs (created by jobs)
    ├── run_000/
    ├── run_001/
    ├── run_002/
    └── run_003/
```

**Tool creates mapping file** (configs/run_mapping.yaml):
```yaml
# Maps run IDs to parameter combinations
runs:
  - id: run_000
    params: {lora_rank: 8}
  - id: run_001
    params: {lora_rank: 16}
  - id: run_002
    params: {lora_rank: 32}
  - id: run_003
    params: {lora_rank: 64}
```

---

### Example 2: Two-Variable Grid (Phase 1)

**Scientific Setup**:
- Research question: How do LoRA rank and learning rate interact?
- Two variables: lora_rank × learning_rate
- Full factorial grid

```yaml
# experiment_002.yaml

experiment:
  name: "lora_lr_grid"
  framework: "torchtune"

  question: "How do LoRA rank and learning rate interact?"
  researcher: "Bob"
  date: "2025-01-10"
  tags: ["hyperparameter-search", "lora"]

framework_config:
  base_config: "llama3_1/8B_lora_single_device"

variables:
  lora_rank: [8, 16, 32]
  learning_rate: [1e-4, 3e-4, 5e-4]
  # Grid: 3 × 3 = 9 runs

controls:
  lora_alpha: 32
  epochs: 3
  batch_size: 2
  dataset: "data/alpaca.json"
  seed: 42

output:
  configs_dir: "configs/"
  results_dir: "results/"
```

**Generated Outputs**:
```
lora_lr_grid/
├── experiment.yaml
├── configs/
│   ├── train_000.yaml    # rank=8, lr=1e-4
│   ├── train_001.yaml    # rank=8, lr=3e-4
│   ├── train_002.yaml    # rank=8, lr=5e-4
│   ├── train_003.yaml    # rank=16, lr=1e-4
│   └── ... (9 total)
└── results/
    ├── run_000/
    └── ...
```

---

### Example 3: Complete Workflow with Evaluation (Phase 1)

**Scientific Setup**:
- Research question: How does LoRA rank affect both training loss and downstream task performance?
- Independent variable: lora_rank
- Evaluation: Test on multiple benchmarks

```yaml
# experiment_003.yaml

experiment:
  name: "lora_rank_with_eval"
  framework: "torchtune"

  question: "How does LoRA rank affect downstream task performance?"
  hypothesis: "Higher ranks will improve performance but with diminishing returns"
  researcher: "Alice"
  date: "2025-01-09"
  tags: ["lora", "evaluation", "llama3.1"]

framework_config:
  base_config: "llama3_1/8B_lora_single_device"

variables:
  lora_rank: [8, 16, 32, 64]

controls:
  learning_rate: 3e-4
  lora_alpha: 32
  epochs: 3
  batch_size: 2
  seed: 42
  dataset: "data/alpaca.json"

  # Configure HF checkpointer for Inspect AI
  _override:
    checkpointer:
      _component_: torchtune.training.FullModelHFCheckpointer
      checkpoint_dir: ${output_dir}/checkpoints
      output_dir: ${output_dir}/hf_model

evaluation:
  enabled: true
  framework: "inspect"
  checkpoint: "last"
  model_format: "hf"
  tasks_dir: "evals/"

  tasks:
    - name: "mmlu_subset"
      file: "mmlu_subset.py"
      function: "mmlu_subset"
      args:
        subjects: ["math", "physics"]

    - name: "domain_qa"
      file: "domain_qa.py"
      function: "domain_qa"
      args:
        split: "test"

output:
  configs_dir: "configs/"
  results_dir: "results/"
  eval_results_dir: "eval_results/"
```

**User-written evaluation task** (evals/mmlu_subset.py):
```python
"""
MMLU evaluation on math and physics subjects
"""
from inspect_ai import Task, task
from inspect_evals.mmlu import mmlu

@task
def mmlu_subset(subjects=None):
    """Evaluate on MMLU subset."""
    return mmlu(subjects=subjects or ["math", "physics"])
```

**User-written evaluation task** (evals/domain_qa.py):
```python
"""
Domain-specific QA evaluation
"""
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import generate, chain_of_thought
from inspect_ai.scorer import match

@task
def domain_qa(split="test"):
    """Evaluate on domain-specific QA."""
    return Task(
        dataset=json_dataset(f"evals/data/domain_qa_{split}.jsonl"),
        solver=[chain_of_thought(), generate()],
        scorer=match(ignore_case=True)
    )
```

**Generated Project Structure**:
```
lora_rank_with_eval/
├── experiment.yaml           # User-written
│
├── evals/                    # USER-WRITTEN evaluation tasks
│   ├── mmlu_subset.py       # ← User writes
│   ├── domain_qa.py         # ← User writes
│   └── data/
│       └── domain_qa_test.jsonl
│
├── configs/                  # GENERATED by tool
│   ├── train_000.yaml       # ← Tool generates (rank=8)
│   ├── train_001.yaml       # ← Tool generates (rank=16)
│   ├── train_002.yaml       # ← Tool generates (rank=32)
│   ├── train_003.yaml       # ← Tool generates (rank=64)
│   ├── eval_000.py          # ← Tool generates
│   ├── eval_001.py          # ← Tool generates
│   ├── eval_002.py          # ← Tool generates
│   ├── eval_003.py          # ← Tool generates
│   └── run_mapping.yaml
│
├── results/                  # Training outputs
│   ├── run_000/
│   │   ├── checkpoints/
│   │   └── hf_model/        # HF format for Inspect
│   ├── run_001/
│   ├── run_002/
│   └── run_003/
│
└── eval_results/             # Evaluation outputs
    ├── run_000/
    │   ├── mmlu_subset/
    │   └── domain_qa/
    ├── run_001/
    ├── run_002/
    └── run_003/
```

**Generated eval script example** (configs/eval_000.py):
```python
#!/usr/bin/env python3
"""
Evaluation script for lora_rank_with_eval run_000
Generated from experiment.yaml

Model: results/run_000/hf_model
Tasks: mmlu_subset, domain_qa
"""

import sys
from pathlib import Path
from inspect_ai import eval

# Add user's evaluation tasks directory to Python path
tasks_dir = Path(__file__).parent.parent / "evals"
sys.path.insert(0, str(tasks_dir))

# Import user-written tasks
from mmlu_subset import mmlu_subset
from domain_qa import domain_qa

# Model checkpoint from training run_000
MODEL_PATH = "hf/results/run_000/hf_model"

# Evaluation tasks (as specified in experiment.yaml)
tasks = [
    mmlu_subset(subjects=["math", "physics"]),
    domain_qa(split="test")
]

# Run evaluation
if __name__ == "__main__":
    print(f"Evaluating model: {MODEL_PATH}")
    print(f"Running {len(tasks)} tasks")

    results = eval(
        tasks,
        model=MODEL_PATH,
        log_dir="eval_results/run_000"
    )

    print(f"✓ Evaluation complete. Results: eval_results/run_000")
```

**Complete Workflow**:

```bash
# 1. User creates experiment definition
cd lora_rank_with_eval/
vim experiment.yaml

# 2. User writes evaluation tasks
mkdir -p evals/data
vim evals/mmlu_subset.py
vim evals/domain_qa.py
vim evals/data/domain_qa_test.jsonl

# 3. Generate all configs
cruijff-kit generate .
# Output:
#   ✓ Generated 4 training configs (train_000.yaml to train_003.yaml)
#   ✓ Generated 4 evaluation scripts (eval_000.py to eval_003.py)
#   ✓ Validated all evaluation tasks

# 4. Submit training jobs to SLURM
sbatch scripts/train_sweep.sh
# Runs: tune run lora_finetune_single_device --config configs/train_NNN.yaml

# 5. After training completes, submit evaluation jobs
sbatch scripts/eval_sweep.sh
# Runs: python configs/eval_NNN.py

# 6. Analyze results
python analyze_results.py eval_results/
```

**Key Points**:
- User writes experiment.yaml AND evaluation tasks (evals/*.py)
- Tool generates training configs AND evaluation scripts
- Training and evaluation run as separate SLURM jobs
- Evaluation uses last checkpoint from training (in HF format)

---

## Mapping to Builder API

### Overview

YAML parameters map to `TorchtuneConfigBuilder` methods:

```yaml
controls:
  learning_rate: 3e-4  # Maps to builder.with_learning_rate(3e-4)
```

The tool translates experiment.yaml → builder API calls → torchtune configs.

---

### Mapping Rules

**Rule 1: Direct 1:1 Mapping** (Preferred)

Most parameters have corresponding `with_*()` methods:

| YAML Key | Builder Method | Example |
|----------|----------------|---------|
| `learning_rate` | `with_learning_rate(lr)` | `learning_rate: 3e-4` |
| `epochs` | `with_epochs(epochs)` | `epochs: 3` |
| `batch_size` | `with_batch_size(batch_size)` | `batch_size: 2` |
| `dataset` | `with_dataset(path)` | `dataset: "data/alpaca.json"` |
| `seed` | `with_seed(seed)` | `seed: 42` |
| `packed` | `with_packed(enabled)` | `packed: true` |
| `dataset_template` | `with_dataset_template(template)` | `dataset_template: "torchtune.data.AlpacaInstructTemplate"` |

**Rule 2: Multi-Parameter Methods** (Special Handling)

Some methods take multiple parameters:

```yaml
# with_lora_params(rank, alpha)
controls:
  lora_rank: 16
  lora_alpha: 32

# Tool translates to: builder.with_lora_params(16, 32)
```

**Rule 3: Fallback to override()** (Everything Else)

Parameters without high-level methods use `override()`:

```yaml
controls:
  clip_grad_norm: 1.0  # No with_clip_grad_norm() method

# Tool translates to: builder.override({"clip_grad_norm": 1.0})
```

**Rule 4: Explicit Override** (Complex Cases)

For nested structures, use `_override`:

```yaml
controls:
  learning_rate: 3e-4  # Uses with_learning_rate()

  _override:
    optimizer:
      weight_decay: 0.01
      amsgrad: true

# Tool translates to:
# builder.with_learning_rate(3e-4)
# builder.override({"optimizer": {"weight_decay": 0.01, "amsgrad": True}})
```

---

### Complete Mapping Table

**Phase 1 High-Level Methods** (from Appendix C):

| YAML Key(s) | Builder Method | Parameter Type |
|-------------|----------------|----------------|
| `dataset` | `with_dataset(path)` | Scientific |
| `output_dir` | `with_output_dir(path)` | Engineering |
| `learning_rate` | `with_learning_rate(lr)` | Scientific |
| `epochs` | `with_epochs(epochs)` | Scientific |
| `lora_rank` + `lora_alpha` | `with_lora_params(rank, alpha)` | Scientific (paired) |
| `batch_size` | `with_batch_size(batch_size)` | Engineering |
| `packed` | `with_packed(enabled)` | Engineering |
| `dataset_template` | `with_dataset_template(template)` | Scientific |
| `seed` | `with_seed(seed)` | Scientific |

**Common Override() Parameters** (no high-level method yet):

| YAML Key | Override Path | Example |
|----------|---------------|---------|
| `clip_grad_norm` | `{"clip_grad_norm": value}` | `clip_grad_norm: 1.0` |
| `gradient_accumulation_steps` | `{"gradient_accumulation_steps": value}` | `gradient_accumulation_steps: 4` |
| `max_seq_length` | `{"tokenizer": {"max_seq_len": value}}` | Use `_override` |
| `activation_checkpointing` | `{"enable_activation_checkpointing": value}` | Use `_override` |
| `compile` | `{"compile": value}` | `compile: true` |
| `dtype` | `{"dtype": value}` | `dtype: "bf16"` |

---

### Translation Algorithm

```python
def translate_experiment_to_configs(experiment: Dict) -> List[str]:
    """
    Translate experiment.yaml to torchtune configs.

    Returns: List of config file paths
    """
    # 1. Load base config
    base_config = experiment['framework_config']['base_config']

    # 2. Generate parameter grid
    variables = experiment['variables']
    grid = cartesian_product(variables)  # e.g., 4 ranks × 4 sizes = 16

    # 3. For each combination in grid:
    configs = []
    for combo in grid:
        # Initialize builder
        builder = TorchtuneConfigBuilder(base_config)

        # Apply controls (fixed params)
        for key, value in experiment['controls'].items():
            if key == '_override':
                builder.override(value)
            elif has_high_level_method(key):
                apply_high_level_method(builder, key, value)
            else:
                builder.override({key: value})

        # Apply variable values for this combo
        for var_name, var_value in combo.items():
            if has_high_level_method(var_name):
                apply_high_level_method(builder, var_name, var_value)
            else:
                builder.override({var_name: var_value})

        # Handle special cases (lora_rank + lora_alpha pairing)
        if 'lora_rank' in combo or 'lora_rank' in experiment['controls']:
            rank = combo.get('lora_rank') or experiment['controls']['lora_rank']
            alpha = combo.get('lora_alpha') or experiment['controls']['lora_alpha']
            builder.with_lora_params(rank, alpha)

        # Generate config filename
        filename = generate_filename(combo, experiment['output']['naming_template'])

        # Save config
        config_path = f"configs/{filename}.yaml"
        builder.save(config_path)
        configs.append(config_path)

    return configs
```

---

## Validation Rules

### Schema Validation

**Required Fields**:
- `experiment.name` (string, non-empty)
- `experiment.framework` (string, must be "torchtune" in Phase 1)
- `framework_config.base_config` (string, non-empty)
- `output.base_dir` (string, non-empty)

**Optional Fields**:
- `experiment.question`, `hypothesis`, `researcher`, `date`, `tags`, `notes`
- `output.naming_template`, `eval_results_dir`
- `evaluation` (entire section optional)

**Type Validation**:
- `variables`: Dict[str, List[Any]]
- `controls`: Dict[str, Any]
- `tags`: List[str]
- `evaluation.enabled`: bool
- `evaluation.tasks`: List[Dict]

---

### Semantic Validation (Q11, Q15)

**Variable Count**:
- ⚠️ Warn if > 2 variables (grid explosion risk)
- ❌ Error if > 3 variables (probably a mistake)

**Grid Size** (Q11):
- ⚠️ Warn if total combinations ≥ 100
- 🛑 Require confirmation to proceed (prevent accidental large sweeps)

**Parameter Conflicts**:
- ❌ Error if same parameter in both `variables` and `controls`

**Multi-Parameter Methods** (Q15):
- ❌ Error if `lora_rank` specified without `lora_alpha` (must be paired)
- ❌ Error if `lora_alpha` specified without `lora_rank`
- ✓ Allow in different sections (rank in variables, alpha in controls)
- ✓ Allow both in variables (cartesian product)

**File References** (Q11 - Strict Validation):
- ❌ Error if dataset file doesn't exist
- ❌ Error if base config doesn't exist (with fuzzy match suggestions)

**Naming Template**:
- ❌ Error if template references non-existent variable
- ⚠️ Warn if template produces duplicate names

---

### Evaluation Validation

**If `evaluation.enabled = true`**:

**Required Fields**:
- ✅ `evaluation.framework` must be "inspect" (Phase 1)
- ✅ `evaluation.tasks_dir` must be specified
- ✅ `evaluation.tasks` must have at least one task

**Directory Validation**:
- ❌ Error if `tasks_dir` doesn't exist
- **Message**: `"Evaluation tasks directory not found: {tasks_dir}. Create the directory and add your evaluation tasks."`

**Task File Validation** (for each task):
- ❌ Error if task file doesn't exist
- ❌ Error if function not found in file
- ❌ Error if function missing `@task` decorator

**Task Arguments Validation**:
- ⚠️ Warn if `args` provided but function doesn't accept kwargs
- Tool doesn't validate argument types (user's responsibility)

**Model Format Validation**:
- ✅ Check that `model_format` is "hf" (Phase 1 only)
- ⚠️ Warn if HF checkpointer not configured in controls/override

**Checkpoint Validation**:
- ✅ Check that `checkpoint` is "last" (Phase 1 only)

---

### Example Validation Errors

```yaml
# ERROR: same param in variables and controls
variables:
  learning_rate: [1e-5, 3e-5]
controls:
  learning_rate: 5e-5  # Conflict!
```

```yaml
# ERROR: lora_rank without lora_alpha
controls:
  lora_rank: 16  # Missing lora_alpha!
```

```yaml
# WARNING: Large grid
variables:
  lora_rank: [8, 16, 32, 64, 128]        # 5 values
  learning_rate: [1e-5, 3e-5, 5e-5, 1e-4]  # 4 values
  batch_size: [1, 2, 4, 8]                # 4 values
  # Total: 5 × 4 × 4 = 80 runs ⚠️ Large sweep!
```

---

## Error Handling

### User-Facing Errors

**Invalid YAML Syntax**:
```
Error: Invalid YAML syntax in experiment_001.yaml:
  Line 15: Expected ':', got ']'
```

**Missing Required Field**:
```
Error: Missing required field 'experiment.name' in experiment_001.yaml
```

**Invalid Framework**:
```
Error: Unsupported framework 'huggingface' in experiment_001.yaml
  Phase 1 only supports framework: "torchtune"
  Supported values: ["torchtune"]
```

**Parameter Conflict**:
```
Error: Parameter 'learning_rate' appears in both variables and controls
  Remove from one section to resolve conflict
```

**Large Grid Warning**:
```
Warning: Experiment will generate 80 configs (5 × 4 × 4)
  This is a large sweep. Consider:
  - Reducing number of variables
  - Using fewer values per variable
  - Running a coarse sweep first

  Continue? [y/N]
```

---

### Evaluation Errors

**Tasks Directory Not Found**:
```
Error: Evaluation tasks directory not found: evals/
  Referenced in: evaluation.tasks_dir

  Create the directory and add your evaluation tasks:
    mkdir -p evals/
    vim evals/my_eval.py
```

**Task File Not Found**:
```
Error: Evaluation task file not found: evals/domain_qa.py
  Referenced in: evaluation.tasks[0]

  Check the file path relative to tasks_dir.
  Expected path: lora_study/evals/domain_qa.py
```

**Task Function Not Found**:
```
Error: Function 'domain_qa' not found in evals/domain_qa.py
  Referenced in: evaluation.tasks[0].function

  Available functions with @task decorator:
    - other_eval
    - safety_check

  Check the function name in experiment.yaml
```

**Missing @task Decorator**:
```
Error: Function 'domain_qa' in evals/domain_qa.py is missing @task decorator
  Referenced in: evaluation.tasks[0]

  All evaluation functions must use @task decorator.

  Example:
    from inspect_ai import task, Task

    @task
    def domain_qa():
        return Task(...)

  See: https://inspect.aisi.org.uk/
```

**HF Checkpointer Not Configured**:
```
Warning: evaluation.enabled=true but HF checkpointer not configured

  Inspect AI requires HuggingFace-format checkpoints.
  Add to your controls or _override:

    controls:
      _override:
        checkpointer:
          _component_: torchtune.training.FullModelHFCheckpointer
          checkpoint_dir: ${output_dir}/checkpoints
          output_dir: ${output_dir}/hf_model
```

**Invalid Evaluation Framework**:
```
Error: Unsupported evaluation framework: 'eleuther'
  Phase 1 only supports framework: "inspect"
  Supported values: ["inspect"]
```

---

### Resource Validation

**Base Config Not Found** (Q11 - Fuzzy Matching):
```
Error: Base config 'llama3_1/8B_lora' not found

  Did you mean one of these?
    - llama3_1/8B_lora_single_device  (exact match except suffix)
    - llama3_1/8B_qlora_single_device
    - llama3_2/8B_lora_single_device

  Run 'tune ls' to see all available configs
```

**Dataset Not Found** (Q11 - Strict Error):
```
Error: Dataset file not found: data/alpaca_100.json
  Referenced in: controls.dataset

  Check the path or create the file before proceeding
```

**Output Directory Exists** (Q11 - Require --force):
```
Error: Output directory exists: lora_study/configs/
  Generated configs would overwrite existing files

  Use --force to overwrite, or choose a different experiment name
```

---

## Tool Implementation (Phase 1 Simplified)

### CLI Interface

**Single Command** (Phase 1):

```bash
# Generate configs from experiment definition
cruijff-kit generate lora_study/
# → Looks for lora_study/experiment.yaml
# → Creates lora_study/configs/ with run_NNN.yaml files
# → Creates lora_study/configs/run_mapping.yaml
```

**Flags** (Phase 1):

```bash
# Dry run: Show what would be generated without writing files
cruijff-kit generate lora_study/ --dry-run

# Force: Overwrite existing configs
cruijff-kit generate lora_study/ --force
```

**Phase 2+ Commands** (Deferred):
- `validate` - Separate validation command
- `list` - List experiments in directory
- Additional flags: `--output-dir`, `--max-configs`, `--full`

### Basic Validation (Phase 1)

**Always performed before generation**:
1. **Schema validation**: Valid YAML, required fields present, correct types
2. **Parameter conflicts**: No param in both variables and controls
3. **Multi-param methods**: lora_rank + lora_alpha both present
4. **File existence**: base_config exists (via `tune ls`), dataset file exists
5. **Grid size warning**: Warn if > 100 total combinations

**No fuzzy matching** (Phase 1): Just error with "not found" message

**No multi-level validation** (Phase 1): Single validation pass

### Output Files (Phase 1 Simplified)

**Without evaluation**:
```
# After running: cruijff-kit generate lora_study/

lora_study/                     # Self-contained experiment folder
├── experiment.yaml             # Input (user-written)
├── configs/                    # Generated configs
│   ├── train_000.yaml          # Torchtune configs
│   ├── train_001.yaml
│   ├── train_002.yaml
│   ├── ...
│   └── run_mapping.yaml        # Maps run IDs to parameters
└── results/                    # Training outputs (jobs write here)
    ├── run_000/
    ├── run_001/
    └── ...
```

**With evaluation enabled**:
```
# After running: cruijff-kit generate lora_study/

lora_study/
├── experiment.yaml             # User-written
├── evals/                      # User-written evaluation tasks
│   ├── my_eval.py
│   └── data/
├── configs/                    # Generated by tool
│   ├── train_000.yaml          # Training configs
│   ├── train_001.yaml
│   ├── eval_000.py             # Evaluation scripts (NEW)
│   ├── eval_001.py
│   └── run_mapping.yaml
├── results/                    # Training outputs
│   ├── run_000/
│   │   └── hf_model/           # HF checkpoint for evaluation
│   └── run_001/
└── eval_results/               # Evaluation outputs (NEW)
    ├── run_000/
    └── run_001/
```

**No experiment_summary.md** (Phase 1): Deferred to Phase 2+

**No manifest.json** (Phase 1): run_mapping.yaml is sufficient

---

### run_mapping.yaml Format (Phase 1)

Simple mapping from run IDs to parameter combinations:

```yaml
# configs/run_mapping.yaml
experiment_name: "lora_rank_study"
generated_at: "2025-01-09T14:30:00Z"
total_runs: 4

runs:
  - id: "run_000"
    params:
      lora_rank: 8
  - id: "run_001"
    params:
      lora_rank: 16
  - id: "run_002"
    params:
      lora_rank: 32
  - id: "run_003"
    params:
      lora_rank: 64
```

---

### experiment_summary.md Format (Phase 2+ - Deferred)

Human-readable summary of the experiment:

```markdown
# Experiment: lora_sample_size_study

**Generated**: 2025-01-09 14:30:00
**Framework**: torchtune
**Base Config**: llama3_1/8B_lora_single_device

## Research Question
How does fine-tuning performance change as we vary LoRA rank and sample size?

## Hypothesis
Larger LoRA ranks need more samples to show benefit; small datasets may overfit with high rank

## Variables (Independent)
- lora_rank: [8, 16, 32, 64]
- sample_size: [100, 500, 1000, 5000]

Total combinations: 16 runs

## Controls (Fixed Parameters)
- learning_rate: 0.0003
- lora_alpha: 32
- epochs: 3
- batch_size: 2
- seed: 42
- dataset: data/alpaca_{sample_size}.json

## Generated Configs

| Config | lora_rank | sample_size | Output Dir |
|--------|-----------|-------------|------------|
| rank8_n100.yaml | 8 | 100 | results/lora_sample_study/rank8_n100 |
| rank8_n500.yaml | 8 | 500 | results/lora_sample_study/rank8_n500 |
| rank8_n1000.yaml | 8 | 1000 | results/lora_sample_study/rank8_n1000 |
| ... | ... | ... | ... |

## Next Steps

1. Validate configs:
   ```bash
   for config in configs/*.yaml; do
     tune validate $config
   done
   ```

2. Submit to SLURM:
   ```bash
   cruijff-kit submit configs/ --cluster-config slurm.yaml
   ```

## Metadata
- Researcher: Alice
- Tags: lora, sample-efficiency, llama3.1
- Design Date: 2025-01-09
```

---

### manifest.json Format (Phase 2+ - Deferred)

Machine-readable metadata for tooling:

```json
{
  "experiment_name": "lora_sample_size_study",
  "experiment_file": "experiment_001.yaml",
  "generated_at": "2025-01-09T14:30:00Z",
  "framework": "torchtune",
  "base_config": "llama3_1/8B_lora_single_device",
  "num_configs": 16,
  "variables": {
    "lora_rank": [8, 16, 32, 64],
    "sample_size": [100, 500, 1000, 5000]
  },
  "configs": [
    {
      "filename": "rank8_n100.yaml",
      "parameters": {
        "lora_rank": 8,
        "sample_size": 100
      },
      "output_dir": "results/lora_sample_study/rank8_n100"
    }
  ]
}
```

---

## Claude Skill Integration (Phase 2+ - Deferred)

### Phase 1 Approach

**No Claude skill** - Users write experiment.yaml manually in their editor.

**Rationale**:
- Format is simple enough to write by hand (see examples above)
- Reduces Phase 1 scope significantly
- Lets us validate the format with real users before building interactive tool
- Can add skill in Phase 2+ once format is proven

### Phase 2+ Skill Design (Future)

When adding Claude skill:

**Basic workflow**:
1. Ask structured questions about experiment
2. Build experiment.yaml in memory
3. Validate and show preview
4. Write file with user approval
5. Call `cruijff-kit generate`
6. Report results

**Features to include**:
- Interactive question-based design
- Auto-fix common mistakes (sanitize names, add paired params)
- Validation before writing
- Learning from past experiments

See earlier version of this appendix for detailed skill design.

---

## Future Extensions

### Phase 2: Advanced Sweep Patterns

**Random Search**:
```yaml
variables:
  learning_rate:
    type: log_uniform
    min: 1e-6
    max: 1e-3
  lora_rank:
    type: choice
    values: [8, 16, 32, 64, 128]

sampling:
  method: random
  n_samples: 20
  seed: 42
```

**Conditional Parameters**:
```yaml
variables:
  model_size: ["8B", "70B"]

controls:
  batch_size:
    if: model_size == "8B"
    then: 4
    else: 1
```

**Multi-Stage Sweeps**:
```yaml
stages:
  - name: "coarse_search"
    variables:
      learning_rate: [1e-5, 1e-4, 1e-3]

  - name: "fine_search"
    based_on: best_from_stage_1
    variables:
      learning_rate: [3e-5, 5e-5, 7e-5]
```

---

### Phase 3: Multi-Framework Support

```yaml
experiment:
  name: "multi_framework_comparison"
  frameworks: ["torchtune", "huggingface"]  # Run same experiment on both

torchtune:
  base_config: "llama3_1/8B_lora_single_device"

huggingface:
  model_id: "meta-llama/Llama-3.1-8B"
  use_peft: true

variables:
  lora_rank: [8, 16, 32]
```

---

### Phase 4: Advanced Output Organization

**Hierarchical Structure**:
```yaml
output:
  structure: "hierarchical"
  # Creates: results/lora_rank/8/sample_size/100/
```

**Custom Metadata**:
```yaml
output:
  metadata:
    save_git_info: true
    save_environment: true
    custom_fields:
      project: "DARPA-2025"
      grant: "N00014-25-1-2345"
```

---

## Appendix: Complete Schema

```yaml
# JSON Schema for experiment.yaml (simplified)
{
  "type": "object",
  "required": ["experiment", "framework_config", "variables", "controls", "output"],
  "properties": {
    "experiment": {
      "type": "object",
      "required": ["name", "framework"],
      "properties": {
        "name": {"type": "string", "minLength": 1},
        "framework": {"type": "string", "enum": ["torchtune"]},
        "question": {"type": "string"},
        "hypothesis": {"type": "string"},
        "researcher": {"type": "string"},
        "date": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"}
      }
    },
    "framework_config": {
      "type": "object",
      "required": ["base_config"],
      "properties": {
        "base_config": {"type": "string", "minLength": 1}
      }
    },
    "variables": {
      "type": "object",
      "patternProperties": {
        ".*": {
          "type": "array",
          "minItems": 1
        }
      }
    },
    "controls": {
      "type": "object"
    },
    "output": {
      "type": "object",
      "required": ["base_dir"],
      "properties": {
        "base_dir": {"type": "string", "minLength": 1},
        "naming_template": {"type": "string"}
      }
    }
  }
}
```

---

**End of Appendix F**
