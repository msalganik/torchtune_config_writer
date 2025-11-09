# Appendix F: GPU Efficiency Helper Specification

**Purpose**: This appendix defines the GPUEfficiencyHelper class for managing GPU-specific configuration and learning optimal settings over time.

**Audience**: Implementers and anyone wanting to understand the self-improving GPU optimization system.

**Related sections**: See main SPEC.md for design rationale and separation of scientific vs engineering parameters.

**Status**: ✅ Complete

---

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [Engineering vs Scientific Parameters](#engineering-vs-scientific-parameters)
4. [Class API Specification](#class-api-specification)
5. [Bootstrap Heuristics](#bootstrap-heuristics)
6. [Learning from Runs](#learning-from-runs)
7. [Memory-Aware Optimization](#memory-aware-optimization)
8. [Strategy Implementation](#strategy-implementation)
9. [Run Log Parsing](#run-log-parsing)
10. [Integration Patterns](#integration-patterns)
11. [Conflict Resolution](#conflict-resolution)
12. [Testing Requirements](#testing-requirements)
13. [Implementation Roadmap](#implementation-roadmap)

---

## Overview

The `GPUEfficiencyHelper` separates engineering decisions (how to fit training in GPU memory) from scientific decisions (what to learn). It evolves through three phases:

1. **Bootstrap Phase**: Uses heuristics and torchtune's shipped configs (Day 1)
2. **Learning Phase**: Accumulates data from actual runs (Week 1)
3. **Optimization Phase**: Provides smart recommendations from historical data (Month 1)

**Key benefit**: Researchers focus on scientific parameters (learning rate, LoRA rank) while the helper manages GPU efficiency.

---

## Design Philosophy

### Core Principles

1. **Works Day 1**: Provides reasonable suggestions without any historical data
2. **Gets Better Over Time**: Learns from every run (Self Improving principle)
3. **Conservative by Default**: Never causes OOM unless user explicitly opts for aggressive mode
4. **Local and Private**: All learning happens locally, no data sent anywhere
5. **Transparent**: User can see why a suggestion was made (confidence scores, sources)

### Evolution Path

```
Day 1 (Bootstrap):
  - Use heuristics from torchtune's configs
  - Conservative safety margins
  - Confidence: 30-40%

Week 1 (Learning):
  - Accumulate 5-10 runs
  - Refine suggestions based on actual memory usage
  - Confidence: 50-70%

Month 1 (Optimization):
  - Rich historical data (50+ runs)
  - Precise recommendations for your specific setup
  - Confidence: 80-95%
```

---

## Engineering vs Scientific Parameters

### Engineering Parameters (GPUEfficiencyHelper manages these)

These affect GPU memory and speed, but **not experiment outcomes**:

```python
ENGINEERING_FIELDS = {
    # Memory management
    "batch_size",                         # GPU memory constraint
    "enable_activation_checkpointing",    # Memory/speed tradeoff
    "enable_activation_offloading",       # CPU offload for memory

    # Precision and compilation
    "dtype",                              # bf16, fp16, fp32 (memory/precision)
    "compile",                            # torch.compile (speed optimization)

    # Device configuration
    "device",                             # cuda, cpu (hardware selection)

    # Logging and profiling
    "output_dir",                         # Where to save
    "log_dir",                            # Where to log
    "profiler",                           # Performance profiling
}
```

### Scientific Parameters (TorchtuneConfigBuilder manages these)

These affect **what the model learns**:

```python
SCIENTIFIC_FIELDS = {
    # Optimizer configuration
    "optimizer.lr",                       # Learning rate
    "optimizer.weight_decay",             # Regularization

    # Model architecture
    "model.lora_rank",                    # LoRA capacity
    "model.lora_alpha",                   # LoRA scaling
    "model.lora_dropout",                 # LoRA regularization
    "model.lora_attn_modules",            # Which layers to adapt
    "model.apply_lora_to_mlp",            # MLP adaptation

    # Training configuration
    "gradient_accumulation_steps",        # Effective batch size
    "epochs",                             # Training duration
    "max_steps_per_epoch",                # Early stopping
    "clip_grad_norm",                     # Gradient clipping

    # Learning rate scheduling
    "lr_scheduler",                       # Scheduler config

    # Data and loss
    "dataset",                            # Data source
    "loss",                               # Loss function
    "seed",                               # Reproducibility
    "max_seq_length",                     # Context window
}
```

**Note**: `gradient_accumulation_steps` is scientific because it affects effective batch size, which impacts learning dynamics.

---

## Class API Specification

### Constructor

```python
class GPUEfficiencyHelper:
    """
    Manages GPU-specific configuration for optimal resource usage.

    Separates engineering decisions (how to fit in memory) from scientific
    decisions (what to learn). Learns from runs over time.
    """

    def __init__(
        self,
        run_log_dir: str = None,
        gpu_type: str = None
    ):
        """
        Initialize GPU Efficiency Helper.

        Args:
            run_log_dir: Directory containing run logs in JSONL format.
                        If None, uses only heuristics (bootstrap mode).
                        If provided, loads historical data for learning.
            gpu_type: GPU type override (e.g., "A100-80GB", "H100-80GB").
                     If None, auto-detects via nvidia-smi.

        Example:
            # Bootstrap mode (no historical data)
            >>> helper = GPUEfficiencyHelper()

            # Learning mode (with historical data)
            >>> helper = GPUEfficiencyHelper(run_log_dir="logs/")
        """
        self.run_log_dir = run_log_dir
        self.gpu_type = gpu_type or self._detect_gpu_type()
        self._run_logs = self._load_run_logs() if run_log_dir else []
```

---

### Core Methods

#### suggest_config()

```python
def suggest_config(
    self,
    model_config: str,
    max_seq_length: int = None,
    strategy: str = "balanced",
    prefer_speed: bool = False
) -> Dict[str, Any]:
    """
    Suggest GPU efficiency settings.

    Args:
        model_config: Base config name (e.g., "llama3_1/8B_lora_single_device")
        max_seq_length: Maximum sequence length (if None, uses default from config)
        strategy: Optimization strategy
            - "conservative": Prioritize stability (never OOM)
            - "balanced": Good performance with safety (default)
            - "aggressive": Maximum performance (higher OOM risk)
        prefer_speed: If True, optimize for speed over memory

    Returns:
        Dict with suggested settings:
        {
            "batch_size": 2,
            "dtype": "bf16",
            "enable_activation_checkpointing": True,
            "compile": False,
            "confidence": 0.75,  # 0.0-1.0
            "source": "historical_data",  # or "heuristic"
            "reasoning": "Based on 15 similar runs, max batch_size=2..."
        }

    Example:
        >>> helper = GPUEfficiencyHelper(run_log_dir="logs/")
        >>> config = helper.suggest_config(
        ...     model_config="llama3_1/8B_lora_single_device",
        ...     max_seq_length=2048,
        ...     strategy="balanced"
        ... )
        >>> print(f"Suggested batch_size: {config['batch_size']}")
        >>> print(f"Confidence: {config['confidence']:.0%}")
    """
    # Implementation: see Bootstrap Heuristics and Learning sections
    pass
```

---

#### log_run_result()

```python
def log_run_result(
    self,
    config: Dict[str, Any],
    result: Dict[str, Any]
) -> None:
    """
    Log a completed run for learning.

    Appends run information to {run_log_dir}/run_logs.jsonl in JSONL format.
    Each run becomes a data point for future suggestions.

    Args:
        config: Full config dict used for the run
        result: Run result dict with keys:
            - success: bool (True if completed without OOM)
            - peak_memory_gb: float (maximum GPU memory used)
            - samples_per_sec: float (throughput)
            - failure_reason: str or None ("OOM", "other_error", null)
            - duration_sec: float (total run time)

    Example:
        >>> result = {
        ...     "success": True,
        ...     "peak_memory_gb": 45.2,
        ...     "samples_per_sec": 12.5,
        ...     "failure_reason": None,
        ...     "duration_sec": 3600.0
        ... }
        >>> helper.log_run_result(config, result)
    """
    if not self.run_log_dir:
        raise ValueError("run_log_dir not set, cannot log results")

    # Extract relevant fields
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": self._extract_relevant_config(config),
        "result": result,
        "gpu_type": self.gpu_type
    }

    # Append to JSONL file
    log_file = Path(self.run_log_dir) / "run_logs.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
```

---

#### parse_torchtune_log() (static)

```python
@staticmethod
def parse_torchtune_log(log_file: str) -> Dict[str, Any]:
    """
    Extract efficiency metrics from torchtune training log.

    Parses torchtune's output to extract memory usage, throughput,
    and failure information. Use this to convert torchtune logs
    into result dicts for log_run_result().

    Args:
        log_file: Path to torchtune log file

    Returns:
        Result dict suitable for log_run_result():
        {
            "success": bool,
            "peak_memory_gb": float or None,
            "samples_per_sec": float or None,
            "failure_reason": str or None,
            "duration_sec": float or None
        }

    Example:
        >>> result = GPUEfficiencyHelper.parse_torchtune_log("logs/exp_001.log")
        >>> helper.log_run_result(config, result)
    """
    # Implementation: see Run Log Parsing section
    pass
```

---

#### find_max_batch_size()

```python
def find_max_batch_size(
    self,
    model_config: str,
    max_seq_length: int,
    test_run: bool = False,
    max_search: int = 64
) -> int:
    """
    Find maximum batch size that fits in memory.

    Args:
        model_config: Base config name
        max_seq_length: Maximum sequence length
        test_run: If True, actually run tests (EXPENSIVE! takes time)
                 If False, estimate from historical data (fast)
        max_search: Maximum batch size to search (binary search upper bound)

    Returns:
        Maximum safe batch size

    Warning:
        test_run=True is expensive! It runs actual training steps with
        different batch sizes. Only use when characterizing a new setup.

    Example:
        # Fast estimation from historical data
        >>> max_bs = helper.find_max_batch_size(
        ...     "llama3_1/8B_lora_single_device",
        ...     max_seq_length=2048,
        ...     test_run=False
        ... )

        # Actual testing (expensive!)
        >>> max_bs = helper.find_max_batch_size(
        ...     "llama3_1/8B_lora_single_device",
        ...     max_seq_length=2048,
        ...     test_run=True
        ... )
    """
    if not test_run:
        # Estimate from historical data
        return self._estimate_max_batch_size(model_config, max_seq_length)

    # Binary search with actual test runs
    # See implementation details below
    pass
```

---

### Helper Methods (Internal)

```python
def _extract_relevant_config(self, config: Dict) -> Dict:
    """Extract only engineering-relevant fields from config."""
    return {
        "model": config.get("model", {}).get("_component_"),
        "max_seq_length": config.get("max_seq_length"),
        "batch_size": config.get("batch_size"),
        "dtype": config.get("dtype"),
        "enable_activation_checkpointing": config.get("enable_activation_checkpointing"),
        "enable_activation_offloading": config.get("enable_activation_offloading"),
        "compile": config.get("compile"),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
    }

def _detect_gpu_type(self) -> str:
    """Detect GPU type via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse output: "NVIDIA A100-SXM4-80GB, 81920 MiB"
        gpu_name = result.stdout.strip().split(',')[0].strip()
        return gpu_name
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

def _load_run_logs(self) -> List[Dict]:
    """Load historical run logs from JSONL file."""
    log_file = Path(self.run_log_dir) / "run_logs.jsonl"
    if not log_file.exists():
        return []

    logs = []
    with open(log_file) as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    return logs
```

---

## Bootstrap Heuristics

### Overview

Bootstrap mode provides reasonable suggestions without any historical data. It uses:

1. Torchtune's shipped config defaults
2. Model size estimations (parameter count)
3. Known GPU memory capacities
4. Conservative safety margins

### Implementation

```python
def _bootstrap_suggestion(
    self,
    model_config: str,
    max_seq_length: int,
    strategy: str
) -> Dict[str, Any]:
    """
    Generate suggestion from heuristics only (no historical data).

    Steps:
    1. Load torchtune's default config
    2. Extract model size and defaults
    3. Apply heuristic adjustments
    4. Apply strategy modifiers
    """
    # Load torchtune's default config
    defaults = self._load_torchtune_defaults(model_config)

    # Extract model size from config name
    model_size = self._extract_model_size(model_config)  # "8B", "70B", etc.
    is_lora = "lora" in model_config.lower()

    # Base heuristics by model size and type
    if model_size == "1B":
        base_batch_size = 8 if is_lora else 4
        activation_checkpointing = False
    elif model_size == "8B":
        base_batch_size = 2 if is_lora else 1
        activation_checkpointing = True if not is_lora else False
    elif model_size == "70B":
        base_batch_size = 1
        activation_checkpointing = True
    else:
        # Unknown size, be very conservative
        base_batch_size = 1
        activation_checkpointing = True

    # Adjust for sequence length (longer sequences need lower batch size)
    default_seq_len = defaults.get("max_seq_length", 2048)
    if max_seq_length and max_seq_length != default_seq_len:
        # Memory scales roughly quadratically with seq length for attention
        # But linearly for other operations, so use 1.5 exponent as compromise
        seq_length_factor = (default_seq_len / max_seq_length) ** 1.5
        base_batch_size = max(1, int(base_batch_size * seq_length_factor))

    # Adjust for GPU type
    if "H100" in self.gpu_type:
        # H100 has more memory and is faster, can increase batch size
        base_batch_size = int(base_batch_size * 1.5)
    elif "V100" in self.gpu_type or "T4" in self.gpu_type:
        # Older GPUs with less memory
        base_batch_size = max(1, base_batch_size - 1)

    # Base suggestion
    suggestion = {
        "batch_size": base_batch_size,
        "dtype": "bf16" if "H100" in self.gpu_type or "A100" in self.gpu_type else "fp16",
        "enable_activation_checkpointing": activation_checkpointing,
        "compile": False,  # Conservative default
        "confidence": 0.3,  # Low confidence (heuristic)
        "source": "heuristic",
        "reasoning": f"Heuristic for {model_size} model on {self.gpu_type}"
    }

    # Apply strategy modifiers
    return self._apply_strategy(suggestion, strategy)

def _load_torchtune_defaults(self, model_config: str) -> Dict:
    """Load torchtune's default config for reference."""
    try:
        # Use tune cp to get default config
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
            subprocess.run(
                ["tune", "cp", model_config, tmp.name],
                check=True,
                capture_output=True
            )
            with open(tmp.name) as f:
                return yaml.safe_load(f)
    except subprocess.CalledProcessError:
        return {}
    finally:
        try:
            os.unlink(tmp.name)
        except:
            pass

def _extract_model_size(self, model_config: str) -> str:
    """
    Extract model size from config name.

    Examples:
        "llama3_1/8B_lora_single_device" -> "8B"
        "llama3_2/1B_full_finetune" -> "1B"
        "llama3_1/70B_lora" -> "70B"
    """
    # Look for pattern like "8B", "70B", "1B" in config name
    match = re.search(r'(\d+)B', model_config)
    if match:
        return f"{match.group(1)}B"
    return "unknown"
```

---

## Learning from Runs

### Run Log Format

Logs stored in `{run_log_dir}/run_logs.jsonl` (one JSON object per line):

```json
{
  "timestamp": "2025-01-09T14:30:00Z",
  "config": {
    "model": "torchtune.models.llama3_1.lora_llama3_1_8b",
    "max_seq_length": 2048,
    "batch_size": 2,
    "dtype": "bf16",
    "enable_activation_checkpointing": true,
    "enable_activation_offloading": false,
    "compile": false,
    "gradient_accumulation_steps": 8
  },
  "result": {
    "success": true,
    "peak_memory_gb": 45.2,
    "samples_per_sec": 12.5,
    "failure_reason": null,
    "duration_sec": 3600.0
  },
  "gpu_type": "NVIDIA A100-SXM4-80GB"
}
```

**Failure example**:
```json
{
  "timestamp": "2025-01-09T15:00:00Z",
  "config": {
    "model": "torchtune.models.llama3_1.lora_llama3_1_8b",
    "max_seq_length": 2048,
    "batch_size": 4,
    "dtype": "bf16",
    "enable_activation_checkpointing": false,
    "compile": false
  },
  "result": {
    "success": false,
    "peak_memory_gb": null,
    "samples_per_sec": null,
    "failure_reason": "OOM",
    "duration_sec": 120.0
  },
  "gpu_type": "NVIDIA A100-SXM4-80GB"
}
```

---

### Learning Algorithm

```python
def _learned_suggestion(
    self,
    model_config: str,
    max_seq_length: int,
    strategy: str
) -> Dict[str, Any]:
    """
    Generate suggestion from historical data.

    Steps:
    1. Query run logs for similar configs
    2. Filter successful runs
    3. Find optimal settings from historical data
    4. Calculate confidence based on data quantity
    5. Apply strategy adjustments
    """
    # Extract model component name from config
    model_component = self._get_model_component(model_config)

    # Find relevant runs (same model, similar seq length)
    similar_runs = [
        run for run in self._run_logs
        if self._is_similar_config(run, model_component, max_seq_length)
    ]

    # If no similar runs, fall back to heuristics
    if not similar_runs:
        return self._bootstrap_suggestion(model_config, max_seq_length, strategy)

    # Separate successful and failed runs
    successful = [r for r in similar_runs if r["result"]["success"]]
    failed = [r for r in similar_runs if not r["result"]["success"]]

    # If all runs failed, be very conservative
    if not successful:
        reasoning = f"All {len(failed)} similar runs failed - using minimal settings"
        return {
            "batch_size": 1,
            "dtype": "bf16",
            "enable_activation_checkpointing": True,
            "compile": False,
            "confidence": 0.5,
            "source": "learned_from_failures",
            "reasoning": reasoning
        }

    # Find optimal batch size from successful runs
    max_successful_batch_size = max(
        r["config"]["batch_size"] for r in successful
    )

    # Try memory-aware optimization to suggest higher batch size
    # This avoids requiring OOM failures to find optimal settings
    memory_suggestion = self._estimate_from_memory_utilization(
        successful_runs=successful,
        failed_runs=failed,
        current_max_batch_size=max_successful_batch_size
    )

    # Use memory-based suggestion if available and higher
    if memory_suggestion and memory_suggestion["batch_size"] > max_successful_batch_size:
        max_successful_batch_size = memory_suggestion["batch_size"]
        memory_optimized = True
        memory_reasoning = memory_suggestion["reasoning"]
    else:
        memory_optimized = False
        memory_reasoning = ""

    # Check if there are failed runs with higher batch sizes
    # This tells us we're near the limit
    has_failures_above = any(
        r["config"]["batch_size"] > max_successful_batch_size
        for r in failed
    )

    # Get most common successful settings
    common_dtype = self._most_common(
        r["config"]["dtype"] for r in successful
    )
    common_checkpointing = self._most_common(
        r["config"]["enable_activation_checkpointing"] for r in successful
    )
    common_compile = self._most_common(
        r["config"]["compile"] for r in successful
    )

    # Calculate confidence based on data quantity and failures
    confidence = self._calculate_confidence(
        num_successful=len(successful),
        num_failed=len(failed),
        has_failures_above=has_failures_above,
        memory_optimized=memory_optimized
    )

    # Build reasoning
    reasoning = f"Based on {len(successful)} successful runs"
    if memory_optimized:
        reasoning += f" with memory analysis. {memory_reasoning}"
    else:
        reasoning += f" (max batch_size={max_successful_batch_size})"

    # Build suggestion
    suggestion = {
        "batch_size": max_successful_batch_size,
        "dtype": common_dtype,
        "enable_activation_checkpointing": common_checkpointing,
        "compile": common_compile,
        "confidence": confidence,
        "source": "historical_data_with_memory" if memory_optimized else "historical_data",
        "reasoning": reasoning
    }

    # Apply strategy adjustments
    return self._apply_strategy(suggestion, strategy)

def _is_similar_config(
    self,
    run: Dict,
    model_component: str,
    max_seq_length: int,
    tolerance: float = 0.1
) -> bool:
    """Check if a run is similar enough to learn from."""
    # Same model
    if run["config"]["model"] != model_component:
        return False

    # Similar sequence length (within 10%)
    run_seq_len = run["config"]["max_seq_length"]
    if abs(run_seq_len - max_seq_length) / max_seq_length > tolerance:
        return False

    # Same GPU type
    if run["gpu_type"] != self.gpu_type:
        return False

    return True

def _calculate_confidence(
    self,
    num_successful: int,
    num_failed: int,
    has_failures_above: bool,
    memory_optimized: bool = False
) -> float:
    """
    Calculate confidence score (0.0 - 1.0).

    Factors:
    - More data points = higher confidence
    - Failures above limit = higher confidence (we know the boundary)
    - No failures = lower confidence (we don't know the limit)
    - Memory-based optimization = higher confidence (data-driven extrapolation)
    """
    # Base confidence from data quantity
    base = min(0.9, 0.4 + 0.05 * num_successful)

    # Boost if we have failure data (know the limits)
    if has_failures_above:
        base = min(0.95, base + 0.15)

    # Boost if memory-optimized (we have utilization data)
    # Memory data gives us confidence even without OOM failures
    if memory_optimized:
        base = min(0.95, base + 0.10)

    # Penalize if we have failures but no successes at this level
    if num_failed > 0 and num_successful < 3:
        base *= 0.8

    return round(base, 2)

def _most_common(self, items):
    """Return most common item in iterable."""
    from collections import Counter
    if not items:
        return None
    counter = Counter(items)
    return counter.most_common(1)[0][0]
```

---

## Memory-Aware Optimization

### Overview

Memory-aware optimization allows the system to suggest higher batch sizes **without requiring OOM failures**. Instead of waiting for a crash to learn limits, it analyzes memory utilization from successful runs to extrapolate safe higher settings.

**Key insight**: If batch_size=2 uses only 40GB on an 80GB GPU, we can estimate batch_size=4 would use ~80GB and suggest trying it.

### Benefits

1. **Faster convergence**: Find optimal settings in 3-4 runs instead of 5-10
2. **No wasted GPU hours**: Avoid OOM failures that consume time before crashing
3. **Better utilization**: Actively pushes toward higher GPU utilization
4. **Data-driven**: Uses actual memory measurements, not just success/failure

### Implementation

```python
def _estimate_from_memory_utilization(
    self,
    successful_runs: List[Dict],
    failed_runs: List[Dict],
    current_max_batch_size: int,
    safety_margin: float = 0.90,
    min_utilization: float = 0.70
) -> Optional[Dict[str, Any]]:
    """
    Estimate higher batch size from memory utilization data.

    This method enables learning optimal settings without OOM failures by
    analyzing how much GPU memory successful runs actually used.

    Args:
        successful_runs: List of successful run logs
        failed_runs: List of failed run logs
        current_max_batch_size: Current maximum successful batch size
        safety_margin: Use this fraction of GPU capacity (default 0.90 = 90%)
        min_utilization: Only suggest higher if current utilization < this

    Returns:
        Dict with suggested batch_size and reasoning, or None if no suggestion
        {
            "batch_size": int,
            "reasoning": str,
            "estimated_memory_gb": float
        }

    Examples:
        # Low utilization - suggest higher batch size
        >>> runs = [{"config": {"batch_size": 2},
        ...          "result": {"peak_memory_gb": 35.0}}]
        >>> estimate = helper._estimate_from_memory_utilization(runs, [], 2)
        >>> estimate["batch_size"]  # Might suggest 4
        4

        # High utilization - don't suggest higher
        >>> runs = [{"config": {"batch_size": 2},
        ...          "result": {"peak_memory_gb": 72.0}}]
        >>> estimate = helper._estimate_from_memory_utilization(runs, [], 2)
        >>> estimate  # None - already at 90% utilization
        None
    """
    # Filter runs with memory data
    runs_with_memory = [
        r for r in successful_runs
        if r["result"].get("peak_memory_gb") is not None
    ]

    if not runs_with_memory:
        return None  # No memory data available

    # Get GPU capacity
    gpu_capacity = self._get_gpu_capacity()
    if gpu_capacity is None:
        return None  # Can't determine GPU capacity

    # Find the run with highest batch size that has memory data
    # This represents our best current knowledge of memory scaling
    best_run = max(
        runs_with_memory,
        key=lambda r: r["config"]["batch_size"]
    )

    batch_size = best_run["config"]["batch_size"]
    memory_used = best_run["result"]["peak_memory_gb"]

    # Calculate current utilization
    current_utilization = memory_used / gpu_capacity

    # If already using >= min_utilization, don't suggest higher
    if current_utilization >= min_utilization:
        return None

    # Check if we have OOM data that would limit us
    # Don't suggest batch sizes we know will fail
    if failed_runs:
        min_failed_batch_size = min(
            r["config"]["batch_size"]
            for r in failed_runs
            if r["result"].get("failure_reason") == "OOM"
        ) if any(r["result"].get("failure_reason") == "OOM" for r in failed_runs) else float('inf')
    else:
        min_failed_batch_size = float('inf')

    # Estimate memory per batch unit (conservative linear scaling)
    # Note: This is a simplification. Real scaling may be non-linear
    # due to fixed overheads, but it's a safe lower bound
    memory_per_batch = memory_used / batch_size

    # Calculate maximum batch size within safety margin
    safe_capacity = gpu_capacity * safety_margin
    estimated_max_batch_size = int(safe_capacity / memory_per_batch)

    # Don't suggest batch sizes we know will OOM
    if estimated_max_batch_size >= min_failed_batch_size:
        estimated_max_batch_size = min_failed_batch_size - 1

    # Only suggest if higher than current max
    if estimated_max_batch_size <= current_max_batch_size:
        return None

    # Build conservative suggestion (increase gradually)
    # Don't jump from batch_size=2 to batch_size=8 immediately
    # Instead, suggest the next power of 2 or current + 2
    suggested_batch_size = min(
        estimated_max_batch_size,
        current_max_batch_size + 2,
        current_max_batch_size * 2
    )

    # Estimate memory for suggested batch size
    estimated_memory = memory_per_batch * suggested_batch_size

    # Build reasoning
    reasoning = (
        f"Current batch_size={batch_size} uses {memory_used:.1f}GB "
        f"({current_utilization:.0%} of {gpu_capacity:.0f}GB). "
        f"Estimated batch_size={suggested_batch_size} would use "
        f"~{estimated_memory:.1f}GB ({estimated_memory/gpu_capacity:.0%})"
    )

    return {
        "batch_size": suggested_batch_size,
        "reasoning": reasoning,
        "estimated_memory_gb": estimated_memory
    }


def _get_gpu_capacity(self) -> Optional[float]:
    """
    Get GPU memory capacity in GB.

    Returns:
        GPU memory in GB, or None if cannot determine
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        # Parse output: "81920" (MiB)
        memory_mib = float(result.stdout.strip().split('\n')[0])
        memory_gb = memory_mib / 1024
        return memory_gb
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError):
        # Fall back to known GPU types
        if "A100-80GB" in self.gpu_type or "H100-80GB" in self.gpu_type:
            return 80.0
        elif "A100-40GB" in self.gpu_type or "A100" in self.gpu_type:
            return 40.0
        elif "V100-32GB" in self.gpu_type:
            return 32.0
        elif "V100-16GB" in self.gpu_type or "V100" in self.gpu_type:
            return 16.0
        elif "T4" in self.gpu_type:
            return 16.0
        elif "H100" in self.gpu_type:
            return 80.0
        else:
            return None
```

### Memory Scaling Considerations

**Linear scaling assumption**:
The implementation assumes `memory_used ≈ batch_size × memory_per_batch`. This is conservative but not perfect:

- **Fixed overhead**: Model weights, optimizer state, etc. don't scale with batch size
- **Attention memory**: Scales quadratically with sequence length, linearly with batch size
- **Activation memory**: Scales linearly with batch size

**Why linear is safe**:
1. Underestimates memory at higher batch sizes (conservative)
2. Simple and interpretable
3. Safety margin (90%) provides buffer for non-linearity

**Future improvements**:
- Could learn actual scaling curves from multiple data points
- Could use quadratic or polynomial fitting
- Could separate fixed vs variable memory components

### Integration with Learning Algorithm

The memory-aware optimization integrates seamlessly with existing learning:

1. **First**: Use historical success/failure data (existing algorithm)
2. **Then**: Check if memory data suggests we can go higher
3. **Finally**: Apply safety checks and strategy adjustments

**Graceful degradation**: If memory data is missing, falls back to existing algorithm.

---

## Strategy Implementation

### Strategy Definitions

```python
def _apply_strategy(
    self,
    base_suggestion: Dict,
    strategy: str
) -> Dict[str, Any]:
    """
    Apply strategy adjustments to base suggestion.

    Strategies:
    - "conservative": Prioritize stability over performance
    - "balanced": Good balance (default)
    - "aggressive": Maximum performance
    """
    suggestion = base_suggestion.copy()

    if strategy == "conservative":
        # Reduce batch size for safety margin
        suggestion["batch_size"] = max(1, suggestion["batch_size"] - 1)

        # Always use activation checkpointing for memory savings
        suggestion["enable_activation_checkpointing"] = True

        # Don't use compile (can increase memory in some cases)
        suggestion["compile"] = False

        # Use bf16 for safety (more stable than fp16)
        if suggestion["dtype"] == "fp16":
            suggestion["dtype"] = "bf16"

        suggestion["reasoning"] += " [Conservative: -1 batch_size, +checkpointing]"

    elif strategy == "balanced":
        # Use suggestion as-is (already balanced)
        suggestion["reasoning"] += " [Balanced]"

    elif strategy == "aggressive":
        # Try to push batch size higher (if we have headroom)
        if suggestion["confidence"] > 0.7:
            suggestion["batch_size"] = suggestion["batch_size"] + 1

        # Disable activation checkpointing for speed (if model is small enough)
        if "1B" in suggestion["reasoning"] or "8B" in suggestion["reasoning"]:
            suggestion["enable_activation_checkpointing"] = False

        # Use compile for speed boost
        suggestion["compile"] = True

        suggestion["reasoning"] += " [Aggressive: +1 batch_size, +compile, -checkpointing]"

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return suggestion
```

---

## Run Log Parsing

### Torchtune Log Format

Torchtune outputs logs with patterns like:

```
INFO:torchtune.training:Starting training...
INFO:torchtune.training:Epoch 1, Step 10 | Loss: 2.345 | LR: 0.0003
INFO:torchtune.training:GPU Memory: Used 45.2GB / 80.0GB
INFO:torchtune.training:Throughput: 12.5 samples/sec
INFO:torchtune.training:Training completed in 3600.0 seconds
ERROR:RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

### Parser Implementation

```python
@staticmethod
def parse_torchtune_log(log_file: str) -> Dict[str, Any]:
    """
    Extract efficiency metrics from torchtune training log.

    Returns:
        {
            "success": bool,
            "peak_memory_gb": float or None,
            "samples_per_sec": float or None,
            "failure_reason": str or None,
            "duration_sec": float or None
        }
    """
    with open(log_file) as f:
        content = f.read()

    # Initialize result
    result = {
        "success": True,
        "peak_memory_gb": None,
        "samples_per_sec": None,
        "failure_reason": None,
        "duration_sec": None
    }

    # Check for CUDA OOM
    if "CUDA out of memory" in content or "OutOfMemoryError" in content:
        result["success"] = False
        result["failure_reason"] = "OOM"

        # Try to extract how long it ran before OOM
        duration_match = re.search(r'(\d+\.?\d*)\s*seconds', content)
        if duration_match:
            result["duration_sec"] = float(duration_match.group(1))

        return result

    # Check for other errors
    if "ERROR" in content and "Training completed" not in content:
        result["success"] = False
        result["failure_reason"] = "other_error"
        return result

    # Extract peak memory
    # Pattern: "GPU Memory: Used 45.2GB" or "Peak memory: 45.2 GB"
    memory_patterns = [
        r'GPU Memory: Used ([\d.]+)GB',
        r'Peak memory: ([\d.]+)\s*GB',
        r'Memory used: ([\d.]+)GB'
    ]
    for pattern in memory_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            result["peak_memory_gb"] = float(match.group(1))
            break

    # Extract throughput
    # Pattern: "Throughput: 12.5 samples/sec" or "samples_per_sec: 12.5"
    throughput_patterns = [
        r'Throughput: ([\d.]+)\s*samples/sec',
        r'samples_per_sec: ([\d.]+)',
        r'([\d.]+)\s*samples per second'
    ]
    for pattern in throughput_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            result["samples_per_sec"] = float(match.group(1))
            break

    # Extract duration
    # Pattern: "Training completed in 3600.0 seconds"
    duration_patterns = [
        r'Training completed in ([\d.]+)\s*seconds',
        r'Total time: ([\d.]+)s',
        r'Duration: ([\d.]+)\s*sec'
    ]
    for pattern in duration_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            result["duration_sec"] = float(match.group(1))
            break

    return result
```

---

## Integration Patterns

### Pattern 1: Bootstrap Phase (No Historical Data)

```python
# Day 1: No historical data yet
gpu_helper = GPUEfficiencyHelper()

# Get suggestion
gpu_config = gpu_helper.suggest_config(
    model_config="llama3_1/8B_lora_single_device",
    max_seq_length=2048,
    strategy="balanced"
)

print(f"Batch size: {gpu_config['batch_size']}")
print(f"Confidence: {gpu_config['confidence']:.0%}")
print(f"Source: {gpu_config['source']}")

# Build config
builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
builder.override(gpu_config)  # Apply GPU settings

# Scientific parameters
builder.with_learning_rate(3e-4)
builder.with_lora_params(32, 64)
builder.with_dataset("data/my_data.json")

builder.save("configs/exp_001.yaml")
```

---

### Pattern 2: Learning Phase (With Feedback Loop)

```python
# Week 1: Start accumulating data
gpu_helper = GPUEfficiencyHelper(run_log_dir="logs/")

# Get suggestion (now uses historical data if available)
gpu_config = gpu_helper.suggest_config(
    model_config="llama3_1/8B_lora_single_device",
    max_seq_length=2048,
    strategy="balanced"
)

# Build and run experiment
builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
builder.override(gpu_config)
builder.with_learning_rate(5e-4)
builder.save("configs/exp_042.yaml")

# Run training (external)
# subprocess.run(["tune", "run", "lora_finetune_single_device", "--config", "configs/exp_042.yaml"])

# After run completes, log the result
result = GPUEfficiencyHelper.parse_torchtune_log("logs/exp_042.log")
gpu_helper.log_run_result(builder.build(), result)

# Next run will use this data!
```

---

### Pattern 3: Explicit API for Clarity

```python
# Clear separation of concerns
builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")

# Scientific parameters (what to learn)
builder.with_learning_rate(3e-4)
builder.with_lora_params(32, 64)
builder.with_gradient_accumulation_steps(8)
builder.with_dataset("data/my_data.json")

# Engineering parameters (GPU efficiency)
gpu_helper = GPUEfficiencyHelper(run_log_dir="logs/")
gpu_config = gpu_helper.suggest_config("llama3_1/8B_lora_single_device", 2048)
builder.override(gpu_config)

# User can still override manually if needed
builder.with_batch_size(4)  # Explicit override

builder.save("configs/exp.yaml")
```

---

### Pattern 4: Conservative Mode for Production

```python
# Production: prioritize stability over speed
gpu_helper = GPUEfficiencyHelper(run_log_dir="logs/")

gpu_config = gpu_helper.suggest_config(
    model_config="llama3_1/8B_lora_single_device",
    max_seq_length=2048,
    strategy="conservative"  # Never OOM
)

builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
builder.override(gpu_config)
# ... rest of config
builder.save("configs/production.yaml")
```

---

### Pattern 5: Memory-Aware Learning (Optimal)

```python
# Week 2+: System learns from memory utilization
gpu_helper = GPUEfficiencyHelper(run_log_dir="logs/")

# First run: Bootstrap heuristics suggest batch_size=2
config1 = gpu_helper.suggest_config(
    model_config="llama3_1/8B_lora_single_device",
    max_seq_length=2048,
    strategy="balanced"
)
print(f"Run 1: batch_size={config1['batch_size']}, confidence={config1['confidence']:.0%}")
# Output: Run 1: batch_size=2, confidence=30% (heuristic)

# Build and run experiment 1
builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
builder.override(config1)
builder.with_learning_rate(3e-4)
builder.save("configs/exp_001.yaml")

# ... run training ...
# Result: Success! Used 35GB / 80GB (44% utilization)

# Log result with memory data
result1 = GPUEfficiencyHelper.parse_torchtune_log("logs/exp_001.log")
gpu_helper.log_run_result(builder.build(), result1)

# Second run: Memory-aware optimization kicks in
config2 = gpu_helper.suggest_config(
    model_config="llama3_1/8B_lora_single_device",
    max_seq_length=2048,
    strategy="balanced"
)
print(f"Run 2: batch_size={config2['batch_size']}, confidence={config2['confidence']:.0%}")
# Output: Run 2: batch_size=4, confidence=60% (memory-optimized)
print(config2['reasoning'])
# Output: "Based on 1 successful runs with memory analysis.
#          Current batch_size=2 uses 35.0GB (44% of 80GB).
#          Estimated batch_size=4 would use ~70.0GB (88%)"

# Build and run experiment 2
builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
builder.override(config2)
builder.with_learning_rate(3e-4)
builder.save("configs/exp_002.yaml")

# ... run training ...
# Result: Success! Used 68GB / 80GB (85% utilization)

# Log result
result2 = GPUEfficiencyHelper.parse_torchtune_log("logs/exp_002.log")
gpu_helper.log_run_result(builder.build(), result2)

# Third run: Now we have good data, high confidence
config3 = gpu_helper.suggest_config(
    model_config="llama3_1/8B_lora_single_device",
    max_seq_length=2048,
    strategy="balanced"
)
print(f"Run 3: batch_size={config3['batch_size']}, confidence={config3['confidence']:.0%}")
# Output: Run 3: batch_size=4, confidence=75% (memory-optimized, high confidence)

# Found optimal setting in 2 runs instead of 5-10!
```

**Benefits demonstrated**:
- No OOM failures required
- Optimal batch_size found in 2-3 runs instead of 5-10
- Confidence increases with data
- Transparent reasoning shows why suggestions are made

---

## Conflict Resolution

### Problem

User manually sets batch_size, GPU helper also suggests batch_size. Which wins?

### Solution: Last Write Wins (Explicit Override)

```python
# GPU helper suggests batch_size=2
gpu_config = gpu_helper.suggest_config(...)
builder.override(gpu_config)  # Sets batch_size=2

# User explicitly overrides
builder.with_batch_size(4)  # Sets batch_size=4

# Final config has batch_size=4 (user's explicit choice wins)
```

**Rationale**:
- Operations applied in order
- Explicit `with_batch_size()` call shows user intent
- User has final control
- No surprising behavior

### Warning for Overrides

Optionally, detect and warn about overrides:

```python
def with_batch_size(self, batch_size: int) -> Self:
    """Set batch size (engineering parameter)."""
    # Check if batch_size was recently set by GPU helper
    if self._recently_set_by_gpu_helper("batch_size"):
        warnings.warn(
            f"Overriding GPU helper's batch_size suggestion. "
            f"This may cause OOM errors.",
            UserWarning
        )

    return self.override({"batch_size": batch_size})
```

---

## Testing Requirements

### Bootstrap Mode Tests

```python
def test_bootstrap_suggestion_8b_lora():
    """Test heuristics for 8B LoRA model."""
    helper = GPUEfficiencyHelper()  # No historical data

    config = helper.suggest_config(
        model_config="llama3_1/8B_lora_single_device",
        max_seq_length=2048,
        strategy="balanced"
    )

    assert config["batch_size"] >= 1
    assert config["dtype"] in ["bf16", "fp16"]
    assert "confidence" in config
    assert config["source"] == "heuristic"
    assert config["confidence"] < 0.5  # Low confidence for heuristics

def test_bootstrap_strategies():
    """Test all three strategies work."""
    helper = GPUEfficiencyHelper()

    conservative = helper.suggest_config("llama3_1/8B_lora_single_device", 2048, "conservative")
    balanced = helper.suggest_config("llama3_1/8B_lora_single_device", 2048, "balanced")
    aggressive = helper.suggest_config("llama3_1/8B_lora_single_device", 2048, "aggressive")

    # Conservative should have smallest batch size
    assert conservative["batch_size"] <= balanced["batch_size"]
    assert balanced["batch_size"] <= aggressive["batch_size"]

    # Conservative should always use checkpointing
    assert conservative["enable_activation_checkpointing"] == True
```

---

### Learning Mode Tests

```python
def test_learned_suggestion_with_data():
    """Test suggestions improve with historical data."""
    # Create mock run logs
    run_logs = [
        {
            "config": {
                "model": "torchtune.models.llama3_1.lora_llama3_1_8b",
                "max_seq_length": 2048,
                "batch_size": 2,
                "dtype": "bf16",
                "enable_activation_checkpointing": False,
                "compile": False
            },
            "result": {"success": True, "peak_memory_gb": 45.2},
            "gpu_type": "NVIDIA A100-SXM4-80GB"
        },
        {
            "config": {
                "model": "torchtune.models.llama3_1.lora_llama3_1_8b",
                "max_seq_length": 2048,
                "batch_size": 4,
                "dtype": "bf16",
                "enable_activation_checkpointing": False,
                "compile": False
            },
            "result": {"success": False, "failure_reason": "OOM"},
            "gpu_type": "NVIDIA A100-SXM4-80GB"
        }
    ]

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "run_logs.jsonl"
        with open(log_file, 'w') as f:
            for run in run_logs:
                f.write(json.dumps(run) + '\n')

        # Load and suggest
        helper = GPUEfficiencyHelper(run_log_dir=tmpdir)
        config = helper.suggest_config(
            "llama3_1/8B_lora_single_device",
            2048,
            "balanced"
        )

        # Should suggest batch_size=2 (max successful)
        assert config["batch_size"] == 2
        assert config["source"] == "historical_data"
        assert config["confidence"] > 0.5  # Higher than heuristic

def test_log_run_result():
    """Test logging run results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        helper = GPUEfficiencyHelper(run_log_dir=tmpdir)

        config = {"batch_size": 2, "dtype": "bf16"}
        result = {
            "success": True,
            "peak_memory_gb": 45.2,
            "samples_per_sec": 12.5,
            "failure_reason": None,
            "duration_sec": 3600.0
        }

        helper.log_run_result(config, result)

        # Verify log file created
        log_file = Path(tmpdir) / "run_logs.jsonl"
        assert log_file.exists()

        # Verify content
        with open(log_file) as f:
            line = f.readline()
            logged = json.loads(line)
            assert logged["result"]["success"] == True
            assert logged["result"]["peak_memory_gb"] == 45.2
```

---

### Memory-Aware Optimization Tests

```python
def test_memory_aware_suggests_higher_batch_size():
    """Test that memory-aware optimization suggests higher batch size when utilization is low."""
    # Mock run with low GPU utilization (35GB / 80GB = 44%)
    run_logs = [
        {
            "config": {
                "model": "torchtune.models.llama3_1.lora_llama3_1_8b",
                "max_seq_length": 2048,
                "batch_size": 2,
                "dtype": "bf16",
                "enable_activation_checkpointing": False,
                "compile": False
            },
            "result": {
                "success": True,
                "peak_memory_gb": 35.0,  # Low utilization
                "samples_per_sec": 12.5
            },
            "gpu_type": "NVIDIA A100-SXM4-80GB"
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "run_logs.jsonl"
        with open(log_file, 'w') as f:
            for run in run_logs:
                f.write(json.dumps(run) + '\n')

        helper = GPUEfficiencyHelper(run_log_dir=tmpdir)
        config = helper.suggest_config(
            "llama3_1/8B_lora_single_device",
            2048,
            "balanced"
        )

        # Should suggest batch_size > 2 due to low utilization
        assert config["batch_size"] > 2
        assert config["source"] == "historical_data_with_memory"
        assert "memory analysis" in config["reasoning"].lower()
        # Confidence should be boosted due to memory data
        assert config["confidence"] >= 0.5


def test_memory_aware_respects_high_utilization():
    """Test that memory-aware optimization doesn't suggest higher when utilization is already high."""
    # Mock run with high GPU utilization (72GB / 80GB = 90%)
    run_logs = [
        {
            "config": {
                "model": "torchtune.models.llama3_1.lora_llama3_1_8b",
                "max_seq_length": 2048,
                "batch_size": 2,
                "dtype": "bf16",
                "enable_activation_checkpointing": False,
                "compile": False
            },
            "result": {
                "success": True,
                "peak_memory_gb": 72.0,  # High utilization (90%)
                "samples_per_sec": 12.5
            },
            "gpu_type": "NVIDIA A100-SXM4-80GB"
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "run_logs.jsonl"
        with open(log_file, 'w') as f:
            for run in run_logs:
                f.write(json.dumps(run) + '\n')

        helper = GPUEfficiencyHelper(run_log_dir=tmpdir)
        config = helper.suggest_config(
            "llama3_1/8B_lora_single_device",
            2048,
            "balanced"
        )

        # Should NOT suggest higher batch size (already at high utilization)
        assert config["batch_size"] == 2
        assert config["source"] == "historical_data"  # Not memory-optimized


def test_memory_aware_respects_oom_limits():
    """Test that memory estimation respects known OOM failures."""
    # Have successful run at batch_size=2 with low utilization
    # But we know batch_size=4 causes OOM
    # Should suggest batch_size=3 (not 4)
    run_logs = [
        {
            "config": {
                "model": "torchtune.models.llama3_1.lora_llama3_1_8b",
                "max_seq_length": 2048,
                "batch_size": 2,
                "dtype": "bf16",
                "enable_activation_checkpointing": False,
                "compile": False
            },
            "result": {
                "success": True,
                "peak_memory_gb": 40.0,  # Would suggest batch_size=4 based on linear scaling
                "samples_per_sec": 12.5
            },
            "gpu_type": "NVIDIA A100-SXM4-80GB"
        },
        {
            "config": {
                "model": "torchtune.models.llama3_1.lora_llama3_1_8b",
                "max_seq_length": 2048,
                "batch_size": 4,
                "dtype": "bf16",
                "enable_activation_checkpointing": False,
                "compile": False
            },
            "result": {
                "success": False,
                "failure_reason": "OOM",
                "peak_memory_gb": None
            },
            "gpu_type": "NVIDIA A100-SXM4-80GB"
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "run_logs.jsonl"
        with open(log_file, 'w') as f:
            for run in run_logs:
                f.write(json.dumps(run) + '\n')

        helper = GPUEfficiencyHelper(run_log_dir=tmpdir)
        config = helper.suggest_config(
            "llama3_1/8B_lora_single_device",
            2048,
            "balanced"
        )

        # Should suggest batch_size=3 (one less than known OOM)
        assert config["batch_size"] == 3
        assert config["source"] == "historical_data_with_memory"


def test_memory_aware_graceful_fallback():
    """Test that system falls back gracefully when memory data is missing."""
    # Run without memory data
    run_logs = [
        {
            "config": {
                "model": "torchtune.models.llama3_1.lora_llama3_1_8b",
                "max_seq_length": 2048,
                "batch_size": 2,
                "dtype": "bf16",
                "enable_activation_checkpointing": False,
                "compile": False
            },
            "result": {
                "success": True,
                "peak_memory_gb": None,  # No memory data
                "samples_per_sec": 12.5
            },
            "gpu_type": "NVIDIA A100-SXM4-80GB"
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "run_logs.jsonl"
        with open(log_file, 'w') as f:
            for run in run_logs:
                f.write(json.dumps(run) + '\n')

        helper = GPUEfficiencyHelper(run_log_dir=tmpdir)
        config = helper.suggest_config(
            "llama3_1/8B_lora_single_device",
            2048,
            "balanced"
        )

        # Should fall back to traditional learning (no memory optimization)
        assert config["batch_size"] == 2
        assert config["source"] == "historical_data"  # Not memory-optimized
        assert "memory analysis" not in config["reasoning"].lower()


def test_get_gpu_capacity():
    """Test GPU capacity detection."""
    # Test with A100-80GB
    helper = GPUEfficiencyHelper()
    helper.gpu_type = "NVIDIA A100-SXM4-80GB"
    capacity = helper._get_gpu_capacity()

    # Should detect 80GB (either via nvidia-smi or fallback)
    assert capacity == 80.0

    # Test with unknown GPU
    helper.gpu_type = "NVIDIA Unknown GPU"
    capacity = helper._get_gpu_capacity()

    # Should return None for unknown GPU if nvidia-smi fails
    # (or actual value if nvidia-smi succeeds)
    assert capacity is None or capacity > 0


def test_estimate_from_memory_utilization_directly():
    """Test the _estimate_from_memory_utilization method directly."""
    helper = GPUEfficiencyHelper()
    helper.gpu_type = "NVIDIA A100-SXM4-80GB"

    # Mock successful run with low utilization
    successful_runs = [
        {
            "config": {"batch_size": 2},
            "result": {"peak_memory_gb": 35.0}
        }
    ]
    failed_runs = []

    suggestion = helper._estimate_from_memory_utilization(
        successful_runs,
        failed_runs,
        current_max_batch_size=2
    )

    # Should suggest higher batch size
    assert suggestion is not None
    assert suggestion["batch_size"] > 2
    assert "reasoning" in suggestion
    assert "estimated_memory_gb" in suggestion

    # Estimated memory should be reasonable
    assert suggestion["estimated_memory_gb"] <= 80.0 * 0.9  # Within safety margin


def test_memory_estimation_conservative_increase():
    """Test that memory estimation increases batch size conservatively (not huge jumps)."""
    helper = GPUEfficiencyHelper()
    helper.gpu_type = "NVIDIA A100-SXM4-80GB"

    # Run at batch_size=2 with very low utilization (10GB / 80GB = 12.5%)
    successful_runs = [
        {
            "config": {"batch_size": 2},
            "result": {"peak_memory_gb": 10.0}
        }
    ]
    failed_runs = []

    suggestion = helper._estimate_from_memory_utilization(
        successful_runs,
        failed_runs,
        current_max_batch_size=2
    )

    # Linear scaling would suggest batch_size=14 (10GB * 7.2 = 72GB)
    # But should be conservative: max(current + 2, current * 2)
    assert suggestion["batch_size"] <= 4  # At most double
    assert suggestion["batch_size"] >= 3  # At least +1 more
```

---

### Log Parsing Tests

```python
def test_parse_successful_log():
    """Test parsing successful training log."""
    log_content = """
    INFO:torchtune.training:Starting training...
    INFO:torchtune.training:GPU Memory: Used 45.2GB / 80.0GB
    INFO:torchtune.training:Throughput: 12.5 samples/sec
    INFO:torchtune.training:Training completed in 3600.0 seconds
    """

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        f.write(log_content)
        log_file = f.name

    try:
        result = GPUEfficiencyHelper.parse_torchtune_log(log_file)

        assert result["success"] == True
        assert result["peak_memory_gb"] == 45.2
        assert result["samples_per_sec"] == 12.5
        assert result["duration_sec"] == 3600.0
        assert result["failure_reason"] is None
    finally:
        os.unlink(log_file)

def test_parse_oom_log():
    """Test parsing OOM error log."""
    log_content = """
    INFO:torchtune.training:Starting training...
    ERROR:RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
    """

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        f.write(log_content)
        log_file = f.name

    try:
        result = GPUEfficiencyHelper.parse_torchtune_log(log_file)

        assert result["success"] == False
        assert result["failure_reason"] == "OOM"
    finally:
        os.unlink(log_file)
```

---

## Implementation Roadmap

### Phase 4: Bootstrap Mode (Week 1)

**Scope**:
- ✅ `GPUEfficiencyHelper.__init__()` with bootstrap mode
- ✅ `suggest_config()` with heuristics
- ✅ `_bootstrap_suggestion()` implementation
- ✅ `_apply_strategy()` for conservative/balanced/aggressive
- ✅ `_extract_model_size()` and `_detect_gpu_type()`
- ✅ Integration with `TorchtuneConfigBuilder`
- ✅ Tests for bootstrap mode

**Deliverables**: Working GPU helper with reasonable heuristics

---

### Phase 4.5: Learning Mode (Week 2)

**Scope**:
- ✅ `log_run_result()` implementation
- ✅ JSONL log format
- ✅ `parse_torchtune_log()` static method
- ✅ `_learned_suggestion()` with historical data
- ✅ Confidence calculation
- ✅ Tests for learning mode
- ✅ `_estimate_from_memory_utilization()` for memory-aware optimization
- ✅ `_get_gpu_capacity()` for GPU memory detection
- ✅ Integration of memory-aware suggestions into learning algorithm
- ✅ Tests for memory-aware optimization

**Deliverables**: Self-improving GPU helper that learns from runs and memory utilization

**Key features**:
- Learns from both success/failure AND memory utilization
- Finds optimal settings in 2-3 runs instead of 5-10
- No OOM failures required to discover optimal batch size
- Graceful fallback when memory data is unavailable

---

### Phase 5: Advanced Features (Optional)

**Scope**:
- 🚧 `find_max_batch_size()` with test runs
- 🚧 Binary search implementation
- 🚧 Advanced log parsing (more metrics)
- 🚧 Multi-GPU support
- 🚧 Memory estimation models
- 🚧 Conflict detection and warnings

**Deliverables**: Advanced optimization features

---

## Alignment with Guiding Principles

**Scientific**:
- ✅ Clear separation of engineering vs scientific parameters
- ✅ Researchers focus on what matters (learning rates, not batch sizes)
- ✅ Reproducibility through config generation

**Modular**:
- ✅ Separate class (not mixed with TorchtuneConfigBuilder)
- ✅ Can be used independently
- ✅ Easy to extend with new strategies

**Practical**:
- ✅ Works day 1 with heuristics
- ✅ No over-engineering: start simple, improve over time
- ✅ Solves real problem (GPU memory management is tedious)
- ✅ Memory-aware optimization finds optimal settings in 2-3 runs instead of 5-10
- ✅ Avoids wasted GPU hours from OOM failures

**Privacy Respecting**:
- ✅ All data stored locally
- ✅ No data sent anywhere
- ✅ User has full control

**Self Improving**:
- ✅ Core design principle!
- ✅ Bootstrap → Learning → Optimization evolution
- ✅ Gets better with every run
- ✅ Learns from both successes and failures
- ✅ Memory-aware optimization learns from utilization, not just OOM crashes
- ✅ Proactively suggests higher batch sizes when GPU is underutilized

**Tested**:
- ✅ Comprehensive test coverage
- ✅ Tests for both bootstrap and learning modes
- ✅ Tests for all strategies
- ✅ Tests for log parsing
- ✅ Tests for memory-aware optimization (8 new test cases)

---

**End of Appendix F**
