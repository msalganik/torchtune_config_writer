# cruijff_kit_v2 - Key Definitions

**Status**: Draft for Review
**Date**: 2025-11-11
**Purpose**: Establish clear, consistent terminology for the project

---

## Core Concepts

### Run

**Definition**: A single computational operation that executes one tool/framework to completion.

**Examples**:
- Training a model with torchtune (one training job)
- Evaluating a model with Inspect AI (one evaluation)
- Optimizing prompts with DSPy (one optimization)

### Experiment

**Definition**: A collection of related runs that address a research question.

**Examples**:
- "Does fine-tuning LLaMA 3 on medical data improve medical QA accuracy?"
- "How does LoRA rank affect model performance on capitalization tasks?"

**Characteristics**:
- Has multiple runs (train, eval, baseline, ablations)
- Has a research question or hypothesis
- Produces scientific findings (papers, reports, insights)
- May span days or weeks
- Has shared context (dataset, model family, research goal)

### Framework

**Definition**: A software framework that performs a specific ML operation.

**Supported in v1.0**:
- **torchtune**: Fine-tuning LLMs with LoRA or full fine-tuning
- **Inspect AI**: Evaluating LLMs on benchmarks or custom tasks
- **DSPy**: Optimizing prompts and demonstrations

**Characteristics**:
- Has its own configuration format (e.g., torchtune YAML)
- Has its own execution method (e.g., `tune run lora_finetune_single_device`)
- Produces specific outputs (checkpoints, metrics, logs)

### Pipeline

**Definition**: A sequence of dependent runs where later runs use outputs from earlier runs.

**Examples**:
- Train → Evaluate: Fine-tune model, then eval on benchmark
- Train → Eval Baseline → Eval Fine-tuned: Compare before/after
- DSPy Optimize → Inspect Eval: Optimize prompts, then evaluate

**Characteristics**:
- Has 2+ runs with dependencies
- Later runs reference earlier runs' outputs (e.g., checkpoint paths)
- Order matters (can't eval before training completes)

### Config (Configuration)

**Definition**: Instructions and parameters that tell a tool how to execute a run.

**Examples**:
- torchtune config: model, dataset, LoRA params, optimizer, training params
- Inspect config: task, model, dataset, generation params
- DSPy config: program, LM, teleprompter, datasets

**Format**: Declarative data (not executable code)
- YAML files (torchtune, most common)
- JSON files
- TOML files
- Command-line arguments
- Environment variables
- Tool-specific configuration formats

### Metadata

**Definition**: Information about runs that answers questions beyond what the run itself produces.

**Purpose**: Enables understanding, tracking, and learning from runs
- **Who/What/When/Where**: What was configured, when was it run, what resources were used
- **Reproducibility**: Capture enough to recreate the run or understand why it succeeded/failed
- **Resource Estimation**: "How long did similar runs take? What GPU was used?"
- **Organization**: Group and find related runs (projects, experiments, pipelines)
- **Analysis**: Query across runs ("show all failed runs", "compare metrics for lora_rank=16")

**Some examples of metadata**:
- Configuration parameters (what was requested)
- Execution details (what actually happened)
- Environment (Python version, framework versions, compute resources)
- Results (metrics, checkpoints, errors)
- Relationships (experiment grouping, pipeline dependencies)
- Context (descriptions, tags, external tracking links)

### Project

**Definition**: A high-level grouping of experiments and runs, typically corresponding to a research effort or publication.

**Examples**:
- "llama3_medical_qa" - all work on medical QA with LLaMA 3
- "capitalization_study" - generalization experiments on capitalization

**Characteristics**:
- Contains multiple experiments
- May span months
- Has shared goals, datasets, models
- Organizational unit for human understanding
