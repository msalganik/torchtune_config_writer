# LoRA Rank Study with Evaluation - Complete Example

This example demonstrates the **complete end-to-end workflow** for designing an experiment with training and evaluation using the experiment definition framework.

## Overview

**Research Question**: How does LoRA rank affect downstream task performance?

**Approach**:
- Sweep over 4 LoRA ranks: [8, 16, 32, 64]
- Train 4 models with identical hyperparameters except rank
- Evaluate each trained model on multiple benchmarks

## Files in This Example

```
lora_rank_with_eval/
├── README.md                    # This file
├── experiment.yaml              # Experiment definition (YOU write this)
│
└── evals/                       # Evaluation tasks (YOU write these)
    ├── mmlu_subset.py          # Example: wrapping inspect_evals
    ├── domain_qa.py            # Example: custom task from scratch
    └── data/
        └── domain_qa_test.jsonl # Sample dataset
```

## What YOU Write

As a user, you write:

1. **experiment.yaml** - Define your experiment
   - Variables to sweep
   - Fixed controls
   - Evaluation configuration

2. **evals/*.py** - Evaluation tasks using Inspect AI
   - Standard `@task` decorated functions
   - Use inspect_evals or write custom tasks
   - Reference your own datasets

## What the TOOL Generates

When you run `cruijff-kit generate .`, the tool generates:

```
lora_rank_with_eval/
├── configs/
│   ├── train_000.yaml          # Torchtune config (rank=8)
│   ├── train_001.yaml          # Torchtune config (rank=16)
│   ├── train_002.yaml          # Torchtune config (rank=32)
│   ├── train_003.yaml          # Torchtune config (rank=64)
│   ├── eval_000.py             # Eval script for run_000
│   ├── eval_001.py             # Eval script for run_001
│   ├── eval_002.py             # Eval script for run_002
│   ├── eval_003.py             # Eval script for run_003
│   └── run_mapping.yaml        # Maps run IDs to parameters
│
├── results/                     # Training outputs (created by jobs)
│   ├── run_000/
│   │   └── hf_model/           # HF checkpoint for evaluation
│   ├── run_001/
│   ├── run_002/
│   └── run_003/
│
└── eval_results/                # Evaluation outputs (created by eval jobs)
    ├── run_000/
    ├── run_001/
    ├── run_002/
    └── run_003/
```

## Complete Workflow

### Step 1: Setup

```bash
# Clone or create your experiment directory
cd lora_rank_with_eval/

# Install dependencies
pip install torchtune inspect-ai inspect-evals
```

### Step 2: Write Your Experiment Definition

Edit `experiment.yaml`:
- Define your research question
- Specify variables to sweep
- Configure fixed hyperparameters
- Add evaluation tasks

See the included `experiment.yaml` for a complete example.

### Step 3: Write Evaluation Tasks

Create evaluation tasks in `evals/`:

**Option A: Wrap inspect_evals (easiest)**
```python
# evals/mmlu_subset.py
from inspect_ai import task
from inspect_evals.mmlu import mmlu

@task
def mmlu_subset(subjects=None):
    return mmlu(subjects=subjects or ["math", "physics"])
```

**Option B: Custom task from scratch**
```python
# evals/domain_qa.py
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import match

@task
def domain_qa(split="test"):
    return Task(
        dataset=json_dataset(f"data/domain_qa_{split}.jsonl"),
        solver=[generate()],
        scorer=match()
    )
```

### Step 4: Generate Configs

```bash
# Generate all training configs and evaluation scripts
cruijff-kit generate .

# Output:
#   ✓ Generated 4 training configs (train_000.yaml to train_003.yaml)
#   ✓ Generated 4 evaluation scripts (eval_000.py to eval_003.py)
#   ✓ Validated all evaluation tasks
```

### Step 5: Run Training

**Option A: Run locally (for testing)**
```bash
# Run single training job
tune run lora_finetune_single_device --config configs/train_000.yaml

# Or run all
for config in configs/train_*.yaml; do
    tune run lora_finetune_single_device --config $config
done
```

**Option B: Submit to SLURM (recommended)**
```bash
# Create SLURM script (or use template)
sbatch scripts/train_sweep.sh
```

### Step 6: Run Evaluation

After training completes:

**Option A: Run locally**
```bash
# Run single evaluation
python configs/eval_000.py

# Or run all
for script in configs/eval_*.py; do
    python $script
done
```

**Option B: Submit to SLURM**
```bash
# After training jobs complete
sbatch scripts/eval_sweep.sh
```

### Step 7: Analyze Results

Evaluation results are in `eval_results/`:
```
eval_results/
├── run_000/
│   ├── mmlu_subset/
│   │   └── results.json
│   └── domain_qa/
│       └── results.json
├── run_001/
└── ...
```

Analyze with Inspect AI tools:
```bash
# View logs
inspect view eval_results/

# Or write custom analysis
python analyze_results.py eval_results/
```

## Key Points

### User Responsibilities
- ✅ Write experiment.yaml
- ✅ Write evaluation tasks (evals/*.py)
- ✅ Prepare evaluation datasets (if custom)
- ✅ Configure HF checkpointer in experiment.yaml

### Tool Responsibilities
- ✅ Generate training configs
- ✅ Generate evaluation scripts
- ✅ Validate task files exist
- ✅ Validate @task decorator present
- ✅ Map parameters to configs

### What the Tool Does NOT Do
- ❌ Provide built-in evaluation tasks (you write them)
- ❌ Run training or evaluation (you do that)
- ❌ Analyze results (you do that with Inspect tools)

## Evaluation Requirements

For evaluation to work, you MUST:

1. **Configure HF checkpointer** in experiment.yaml:
   ```yaml
   controls:
     _override:
       checkpointer:
         _component_: torchtune.training.FullModelHFCheckpointer
         checkpoint_dir: ${output_dir}/checkpoints
         output_dir: ${output_dir}/hf_model
   ```

2. **Write evaluation tasks** with `@task` decorator

3. **Install Inspect AI**:
   ```bash
   pip install inspect-ai
   # For pre-built evals (optional):
   pip install inspect-evals
   ```

## Troubleshooting

**Error: "Evaluation tasks directory not found: evals/"**
- Create the directory: `mkdir -p evals/`

**Error: "Function 'domain_qa' not found in evals/domain_qa.py"**
- Check the function name matches experiment.yaml
- Ensure function has `@task` decorator

**Error: "Module 'inspect_ai' not found"**
- Install: `pip install inspect-ai`

**Evaluation fails: "Cannot load model"**
- Ensure HF checkpointer is configured in experiment.yaml
- Check that training completed successfully
- Verify `results/run_NNN/hf_model/` directory exists

## Next Steps

1. **Customize experiment.yaml** for your research question
2. **Write your evaluation tasks** using Inspect AI
3. **Generate and validate** configs before submitting to SLURM
4. **Run experiments** and collect results
5. **Analyze** using Inspect AI tools or custom scripts

## References

- [Torchtune Documentation](https://pytorch.org/torchtune/)
- [Inspect AI Documentation](https://inspect.aisi.org.uk/)
- [Inspect Evals](https://github.com/UKGovernmentBEIS/inspect_evals)
- [Appendix F: Experiment Definition Format](../../appendices/F_experiment_definition.md)
