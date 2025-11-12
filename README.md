# cruijff_kit: Torchtune Config Generation

**Component of cruijff_kit v2 - Research experiment orchestration for LLM fine-tuning**

This component generates torchtune YAML configuration files from experiment definitions, enabling researchers to define parameter sweeps declaratively and generate complete configs for SLURM job submission.

**Status**: 🚧 Phase 1 Specification (Implementation in Progress)

---

## What It Does

Converts a single `experiment.yaml` file into multiple torchtune configs for parameter sweeps:

**Input (experiment.yaml):**
```yaml
experiment:
  name: lora_rank_study
  framework: torchtune
  question: "How does LoRA rank affect performance?"

framework_config:
  base_config: llama3_2/1B_lora_single_device  # Run 'tune ls' to see available configs

variables:
  lora_rank: [8, 16, 32, 64]  # 4 values = 4 configs

controls:
  learning_rate: 3e-4
  epochs: 3
  dataset:
    data_files: /path/to/data.json
```

**Output:**
```
configs/
├── run_000.yaml    # lora_rank=8
├── run_001.yaml    # lora_rank=16
├── run_002.yaml    # lora_rank=32
├── run_003.yaml    # lora_rank=64
└── run_mapping.yaml
```

**Example run_mapping.yaml:**
```yaml
run_000:
  variables:
    lora_rank: 8
run_001:
  variables:
    lora_rank: 16
run_002:
  variables:
    lora_rank: 32
run_003:
  variables:
    lora_rank: 64
```

Each config is a complete, valid torchtune configuration ready for:
```bash
tune run lora_finetune_single_device --config configs/run_000.yaml
```

---

## Why This Exists

**The Problem:**
- Researchers think in terms of **variables** (what I'm testing) and **controls** (what's constant)
- Torchtune needs separate YAML files for each configuration
- Manually creating/editing dozens of configs is error-prone and not reproducible

**The Solution:**
- Define experiments declaratively in `experiment.yaml`
- Tool generates full factorial sweep (cartesian product of variables)
- Start from torchtune's proven recipes or your custom configs
- Validate all configs before submitting to SLURM

**Scientific Benefit:**
- experiment.yaml clearly documents your experimental design
- Six months later, you can see exactly what varied and what didn't
- Reproducible, testable, reviewable

---

## Usage

### Manual Workflow (Human Researchers)

**1. Write experiment.yaml**
```yaml
experiment:
  name: my_experiment
  framework: torchtune

framework_config:
  base_config: llama3_2/1B_lora_single_device  # Run 'tune ls' to see available configs

variables:
  lora_rank: [8, 16, 32]

controls:
  learning_rate: 3e-4
  batch_size: 4
```

**2. Generate configs**
```bash
python -m torchtune_config_writer generate experiment.yaml

# Output:
# ✓ Loaded base config: llama3_2/1B_lora_single_device
# ✓ Generated 3 configs in outputs/my_experiment_20250112_143022/configs/
# ✓ Created run_mapping.yaml
```

**3. Submit to SLURM**
```bash
# Submit individual jobs
sbatch submit_run_000.sh

# Or use SLURM job arrays
sbatch --array=0-2 submit_array.sh
```

### Skills Workflow (Claude Code Automation)

**Skills use CLI (Phase 1):**
```python
subprocess.run([
    "python", "-m", "torchtune_config_writer", "generate", "experiment.yaml"
], check=True)
```

**Python API (deferred to Phase 2):**
```python
# Future: Direct Python API planned for Phase 2
# from cruijff_kit.torchtune import generate_configs
```

---

## Key Features

**✅ Phase 1 (Current Scope)**
- Load from torchtune recipes OR custom config files
- Variables vs. controls (mirrors scientific thinking)
- Full factorial sweep (cartesian product)
- Deep merge semantics (override specific fields without rewriting everything)
- Pydantic validation
- Optional: `tune validate` integration

**Note:** Large sweeps (1000+ combinations) may take time to generate and validate. Consider using smaller test sweeps first to verify your configuration.

**🚧 Planned for Phase 2+**
- Evaluation config generation
- SLURM script generation
- Metadata tracking
- Python fluent API enhancements
- Variable substitution in paths

---

## Phase 1 Scope

**What's Included:**
- experiment.yaml → torchtune configs generation
- CLI interface (`python -m torchtune_config_writer generate`)
- Standalone usage (cruijff_kit integration planned for v2)
- Comprehensive spec and test suite

**What's Deferred:**
- Evaluation (Inspect AI integration)
- SLURM orchestration
- Metadata tracking beyond experiment.yaml

See [SPEC.md](SPEC.md) for complete technical specification.

---

## Documentation

- **[SPEC.md](SPEC.md)** - Complete technical specification
- **[Appendix A](appendices/A_merge_semantics.md)** - Detailed merge rules and edge cases
- **[Appendix F](appendices/F_experiment_definition.md)** - Complete experiment.yaml format spec
- **[DEFINITIONS.md](DEFINITIONS.md)** - Terminology reference
- **[REFACTORING_PLAN.md](REFACTORING_PLAN.md)** - Migration rationale from v1
- **[CRUIJFF_KIT_V1_SUMMARY.md](CRUIJFF_KIT_V1_SUMMARY.md)** - v1 architecture analysis
- **[SETUP.md](SETUP.md)** - Environment setup instructions

---

## Prerequisites

- Python 3.9+
- torchtune installed with CLI tools:
  ```bash
  pip install torchtune

  # Verify CLI works:
  tune ls  # Should list available configs
  ```

## Installation (Development)

```bash
# Clone repo
git clone <repo-url>
cd torchtune_config_writer

# Setup environment
./setup.sh
source .venv/bin/activate

# Run tests
pytest

# Install in development mode
pip install -e .
```

**Note:** This component will be integrated into the cruijff_kit monorepo as `cruijff_kit.torchtune`.

---

## Example: LoRA Rank Sweep

**experiment.yaml:**
```yaml
experiment:
  name: lora_rank_study
  framework: torchtune
  question: "How does LoRA rank affect downstream task performance?"
  researcher: "Alice"

framework_config:
  base_config: llama3_2/1B_lora_single_device  # Run 'tune ls' to see available configs

variables:
  lora_rank: [8, 16, 32, 64]

controls:
  learning_rate: 3e-4
  lora_alpha: 16
  epochs: 3
  batch_size: 4
  dataset:
    data_files: /scratch/data/my_data.json
    packed: true
  output_dir: /scratch/results/lora_study
  metric_logger:
    _component_: torchtune.training.metric_logging.WandBLogger
    project: lora_rank_study
    mode: offline
```

**Run:**
```bash
python -m torchtune_config_writer generate experiment.yaml
```

**Result:**
- 4 complete torchtune configs (run_000.yaml through run_003.yaml)
- run_mapping.yaml for traceability
- All configs ready for `tune validate` and SLURM submission

---

## Contributing

This project follows the principles in [CLAUDE.md](CLAUDE.md):
- **Scientific**: Correctness, reproducibility, detailed logging
- **Modular**: Components can be added/changed independently
- **Practical**: Do science, not programming contests
- **Privacy Respecting**: Treat data with care
- **Self Improving**: Learn from earlier experiments
- **Tested**: Write tests alongside development

---

## License

MIT

---

## Questions or Feedback?

- See [SPEC.md](SPEC.md) for detailed technical documentation
- Check [Issues](https://github.com/niznik-dev/cruijff-kit/issues) for known issues
- This is part of cruijff_kit v2 - see main repo for broader context
