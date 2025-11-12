# Appendix F: Experiment Definition Format (Phase 1)

**Purpose**: Define the `experiment.yaml` format for generating torchtune configs from parameter sweeps.

**Status**: Phase 1 - Core functionality only

---

## Overview

Scientists define experiments by specifying:
- **Experiment metadata**: Name, research question, researcher info
- **Framework**: Which tool to use (torchtune, inspect, dspy)
- **Variables**: Parameters to sweep (independent variables)
- **Controls**: Parameters held constant (controlled variables)
- **Base config**: Starting point (torchtune recipe or custom file)

The tool generates one complete torchtune config per variable combination.

**Example:**
```yaml
experiment:
  name: lora_rank_study
  framework: torchtune
  question: "How does LoRA rank affect model performance?"
  researcher: "Alice"

framework_config:
  base_config: llama3_2/1B_lora_single_device

variables:
  lora_rank: [8, 16, 32, 64]

controls:
  learning_rate: 3e-4
  dataset:
    data_files: /path/to/my_data.json
```

Generates 4 configs: one for each `lora_rank` value, all with the same `learning_rate` and `dataset`.

---

## Format Specification

### Required Fields

```yaml
experiment:
  name: "my_experiment"           # REQUIRED: Unique identifier
  framework: "torchtune"          # REQUIRED: Framework (torchtune, inspect, dspy)
  question: "Research question"   # Optional: What you're testing
  hypothesis: "Expected outcome"  # Optional: What you expect
  researcher: "Name"              # Optional: Who's running this
  date: "2025-11-12"             # Optional: When designed
  tags: ["lora", "llama3"]       # Optional: For organization

framework_config:
  # REQUIRED: Exactly one of these
  base_config: "llama3_2/1B_lora_single_device"  # Torchtune recipe name
  # OR
  base_config_file: "/path/to/custom_config.yaml"  # Path to custom config
```

**Experiment Section:**
- **name**: Required. Unique identifier for this experiment
- **framework**: Required. Must be "torchtune" in Phase 1 (will error if set to "inspect", "dspy", or other values). Multi-framework support planned for cruijff_kit v2.
- **question, hypothesis, researcher, date, tags**: Optional metadata for tracking and organization

**Framework Config Section:**

1. **`base_config`**: Use torchtune's built-in recipe
   - Value: Recipe name from `tune ls` (e.g., `"llama3_2/1B_lora_single_device"`)
   - Run `tune ls` to see all available configs
   - Tool uses `tune cp` to get the base config

2. **`base_config_file`**: Use your own custom config file
   - Value: Absolute path to a YAML config file
   - Useful when you've already adapted a config for your GPU/cluster

**You must specify exactly one.** Specifying both will raise a ValidationError.

---

### Variables Section (Parameter Sweep)

```yaml
variables:
  lora_rank: [8, 16, 32, 64]
  learning_rate: [1e-4, 3e-4]
```

**Behavior:**
- Generates **full factorial grid** (cartesian product)
- Example above: 4 × 2 = 8 configs
- Each parameter can be a list of: numbers, strings, booleans
- If variables section is empty or omitted: generates single config with just base + controls

**Supported parameters:** Any valid torchtune config parameter. Common examples:
- `lora_rank`, `lora_alpha`
- `learning_rate` (note: maps to `optimizer.lr` in config)
- `batch_size`
- `epochs`
- Dataset parameters (see nested overrides below)

---

### Controls Section (Fixed Parameters)

```yaml
controls:
  epochs: 3
  batch_size: 2
  seed: 42
```

**Behavior:**
- Applied to ALL generated configs
- Parameter types: numbers, strings, booleans, null, nested dicts

**Simple overrides:**
```yaml
controls:
  learning_rate: 3e-4  # Scalar override
  batch_size: 4
```

**Nested overrides:**
```yaml
controls:
  dataset:
    data_files: /path/to/my_data.json
    packed: true
  optimizer:
    weight_decay: 0.01
  metric_logger:
    _component_: torchtune.training.metric_logging.WandBLogger
    project: my_project
    mode: offline
```

**Merge behavior:**
- Scalar values replace
- Nested dicts merge recursively (see [Appendix A](A_merge_semantics.md) for details)
- Lists replace entirely (no appending)

---

### Output Section (Optional)

```yaml
output:
  experiment_name: "my_experiment"  # Optional: used in folder name
  base_dir: "./outputs"             # Optional: base output directory
```

All fields optional. The tool creates a unique folder structure:
```
outputs/
├── my_experiment_20250112_143022/
│   ├── experiment.yaml      # Copy of source for reproducibility
│   ├── configs/
│   │   ├── run_000.yaml    # Generated configs
│   │   └── ...
│   ├── run_mapping.yaml    # Maps run IDs to variable values
│   └── validation_report.txt  # tune validate results
```

Each experiment gets a unique timestamped folder to prevent collisions.

---

## Complete Examples

### Example 1: Single Variable Sweep (Torchtune Base)

```yaml
experiment:
  name: lora_rank_study
  framework: torchtune
  question: "How does LoRA rank affect model performance?"
  researcher: "Alice"

framework_config:
  base_config: llama3_2/1B_lora_single_device

variables:
  lora_rank: [8, 16, 32, 64]

controls:
  learning_rate: 3e-4
  lora_alpha: 16
  epochs: 3
  batch_size: 4
  dataset:
    data_files: /scratch/data/alpaca.json
    packed: true
  output_dir: /scratch/results/lora_study
```

**Generates:** 4 configs (`configs/run_000.yaml` through `run_003.yaml`)

---

### Example 2: Two-Variable Grid (Custom Base Config)

```yaml
experiment:
  name: lr_batch_sweep
  framework: torchtune
  question: "What's the optimal learning rate and batch size combination?"

framework_config:
  base_config_file: /home/user/configs/my_gpu_optimized.yaml

variables:
  learning_rate: [1e-4, 3e-4, 5e-4]
  batch_size: [2, 4, 8]

controls:
  epochs: 5
  seed: 42
```

**Generates:** 9 configs (3 learning rates × 3 batch sizes)

---

### Example 3: With W&B Logging

```yaml
experiment:
  name: cap_viz_test
  framework: torchtune
  question: "Does capitalization pattern generalize across word lengths?"
  researcher: "Bob"
  tags: ["capitalization", "generalization"]

framework_config:
  base_config: llama3_2/1B_lora_single_device

variables:
  lora_rank: [8, 16]

controls:
  dataset:
    data_files: /path/to/words.json
    field: train
  epochs: 2
  batch_size: 16

  # Override metric logger
  metric_logger:
    _component_: torchtune.training.metric_logging.WandBLogger
    project: capitalization
    mode: offline
    log_dir: ${output_dir}/logs
```

---

## Generated Files

For an experiment with N variable combinations:

```
experiment_name/
├── experiment.yaml           # User-written definition
└── configs/                  # Generated by tool
    ├── run_000.yaml          # Complete torchtune config
    ├── run_001.yaml
    ├── ...
    ├── run_NNN.yaml
    └── run_mapping.yaml      # Maps run IDs to parameter values
```

**run_mapping.yaml format:**
```yaml
runs:
  - id: run_000
    params: {lora_rank: 8, learning_rate: 1e-4}
  - id: run_001
    params: {lora_rank: 8, learning_rate: 3e-4}
  # ...
```

---

## Usage

```bash
# Generate configs from experiment.yaml
python -m torchtune_config_writer generate experiment.yaml

# Output:
# ✓ Loaded base config: llama3_2/1B_lora_single_device
# ✓ Generated 4 configs in configs/
# ✓ Created run_mapping.yaml
```

---

## Parameter Reference

**Common torchtune parameters:**

| Parameter | Type | Example | Notes |
|-----------|------|---------|-------|
| `lora_rank` | int | `8` | LoRA rank |
| `lora_alpha` | int | `16` | Usually 2 × rank |
| `learning_rate` | float | `3e-4` | Maps to `optimizer.lr` |
| `batch_size` | int | `4` | Per-device batch size |
| `epochs` | int | `3` | Training epochs |
| `seed` | int | `42` | Random seed |
| `output_dir` | string | `"/path"` | Checkpoint output |

**Dataset override:**
```yaml
dataset:
  _component_: torchtune.datasets.instruct_dataset
  source: json
  data_files: /path/to/data.json
  field: train
  packed: true
```

See torchtune's config examples for full parameter reference.

---

## Validation

The tool validates:
- ✓ Exactly one of `base_config` or `base_config_file` specified (error if both)
- ✓ Base config exists (validates with `tune cp` test)
- ✓ Framework must be "torchtune" in Phase 1
- ✓ YAML syntax valid
- ✓ Generated configs validated with `tune validate` (post-generation)

---

## Phase 2+ Features (Not Yet Implemented)

- Variable substitution in paths: `"data_{sample_size}.json"`
- Conditional parameters
- Random/grid search strategies
- Evaluation configuration
- Multi-framework support
- Best checkpoint selection

---

## See Also

- [Appendix A: Merge Semantics](A_merge_semantics.md) - How overrides are applied
- [SPEC.md](../SPEC.md) - Overall architecture
- [DEFINITIONS.md](../DEFINITIONS.md) - Terminology
