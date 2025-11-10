"""
MMLU Evaluation Task (Example)

This is a USER-WRITTEN evaluation task that wraps inspect_evals MMLU.
The tool generates scripts that import and run this task.

Requirements:
- Must have @task decorator from inspect_ai
- Must return Task object
- Can accept arguments (passed from experiment.yaml)
"""

from inspect_ai import Task, task
from inspect_evals.mmlu import mmlu


@task
def mmlu_subset(subjects=None):
    """
    Evaluate model on MMLU subset.

    Args:
        subjects: List of MMLU subjects to evaluate on.
                  If None, defaults to ["math", "physics"]

    Returns:
        Task: Inspect AI task object
    """
    # Default to math and physics if not specified
    if subjects is None:
        subjects = ["math", "physics"]

    # Use the inspect_evals MMLU task with custom subjects
    return mmlu(subjects=subjects)


# ==============================================================================
# NOTES FOR USERS
# ==============================================================================
#
# This task wraps inspect_evals.mmlu with a custom subset.
# You can:
# 1. Use inspect_evals tasks directly (import and wrap)
# 2. Write completely custom tasks with your own datasets
# 3. Pass arguments from experiment.yaml
#
# The tool will generate eval scripts that call:
#   mmlu_subset(subjects=["math", "physics"])
#
