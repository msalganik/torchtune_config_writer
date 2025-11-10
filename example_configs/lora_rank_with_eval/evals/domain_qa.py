"""
Domain-Specific QA Evaluation Task (Example)

This is a USER-WRITTEN custom evaluation task.
Demonstrates how to write evaluation tasks from scratch.

Requirements:
- Must have @task decorator from inspect_ai
- Must return Task object
- Define dataset, solver, and scorer
"""

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import generate, chain_of_thought
from inspect_ai.scorer import match


@task
def domain_qa(split="test", max_samples=None):
    """
    Evaluate model on domain-specific QA dataset.

    This is a custom task that uses a custom dataset format.
    The dataset should be in JSONL format with fields:
    - input: The question/prompt
    - target: Expected answer(s)

    Args:
        split: Dataset split to use ("train", "test", "val")
        max_samples: Maximum number of samples to evaluate (None = all)

    Returns:
        Task: Inspect AI task object
    """
    # Load dataset from JSONL file
    # Path is relative to the evaluation task file
    dataset_path = f"data/domain_qa_{split}.jsonl"
    dataset = json_dataset(dataset_path)

    # Limit samples if specified
    if max_samples is not None:
        dataset = dataset[:max_samples]

    # Define the evaluation task
    return Task(
        dataset=dataset,
        solver=[
            chain_of_thought(),  # Use chain-of-thought reasoning
            generate()           # Generate response
        ],
        scorer=match(ignore_case=True)  # Match against target
    )


# ==============================================================================
# DATASET FORMAT
# ==============================================================================
#
# The domain_qa_{split}.jsonl file should contain one JSON object per line:
#
# {"input": "What is the capital of France?", "target": ["Paris", "paris"]}
# {"input": "What is 2+2?", "target": ["4", "four"]}
#
# Fields:
# - input: Question or prompt for the model
# - target: Expected answer(s) - can be string or list of acceptable answers
#
# ==============================================================================
# USAGE IN EXPERIMENT.YAML
# ==============================================================================
#
# evaluation:
#   tasks:
#     - name: "domain_qa"
#       file: "domain_qa.py"
#       function: "domain_qa"
#       args:
#         split: "test"
#         max_samples: 100
#
# The tool will generate eval scripts that call:
#   domain_qa(split="test", max_samples=100)
#
