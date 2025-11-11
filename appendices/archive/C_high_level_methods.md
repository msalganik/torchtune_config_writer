# Appendix C: High-Level Methods Specification

**Purpose**: This appendix defines all high-level convenience methods for common config modifications, organized by category (scientific vs engineering parameters).

**Audience**: Implementers and users wanting to understand available convenience methods.

**Related sections**: See main SPEC.md for architecture and design decisions.

**Status**: ✅ Complete - Ready for Phase 1 implementation

---

## Overview

High-level methods provide convenience wrappers around `override()` for common operations. This improves usability and self-documentation of code.

## Design Principles

1. **Naming**: Use `with_*` prefix for all methods
2. **Return**: Always return `self` for chaining
3. **Validation**: Validate inputs, raise clear errors
4. **Simplicity**: Wrapper around override(), no complex logic
5. **Documentation**: Clear docstrings with examples
6. **Pairing**: Force pairing of scientifically-related parameters (e.g., lora_rank + lora_alpha)

---

## Existing Methods (Phase 1 - Already Specified)

These methods have basic signatures in main SPEC.md:

1. `with_dataset(path: str)` - Change dataset source path
2. `with_output_dir(path: str)` - Change output directory
3. `with_learning_rate(lr: float)` - Change optimizer learning rate
4. `with_batch_size(batch_size: int)` - Change training batch size
5. `with_epochs(epochs: int)` - Change number of training epochs

**Note**: These need full specifications added (TODO: expand like methods below).

---

## New Methods (Phase 1 - Complete Specifications)

### Method 1: with_lora_params()

**Category**: Scientific parameter (affects model architecture)

#### Signature

```python
def with_lora_params(self, rank: int, alpha: int) -> 'TorchtuneConfigBuilder':
    """
    Set LoRA rank and alpha together.

    LoRA (Low-Rank Adaptation) uses low-rank matrices to fine-tune models efficiently.
    These parameters are scientifically related and should be set together:
    - rank: Determines size of adapter matrices (higher = more parameters)
    - alpha: Scaling factor for adapter updates (typical convention: alpha = 2 * rank)

    Args:
        rank: LoRA rank (positive integer, typically 8-64)
        alpha: LoRA alpha (positive integer, typically 2*rank)
               Must be specified explicitly - no default provided

    Returns:
        Self for method chaining

    Raises:
        TypeError: If rank or alpha is not an int
        ValueError: If rank or alpha is <= 0

    Example:
        >>> # Following convention: alpha = 2 * rank
        >>> builder.with_lora_params(32, 32 * 2)

        >>> # Custom ratio
        >>> builder.with_lora_params(32, 48)

        >>> # Parameter sweep
        >>> for rank in [8, 16, 32, 64]:
        ...     builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
        ...     builder.with_lora_params(rank, rank * 2)
        ...     builder.save(f"configs/rank_{rank}.yaml")
    """
```

#### Behavior

**Config paths modified**: `model.lora_rank`, `model.lora_alpha`

**Override call**:
```python
return self.override({
    "model": {
        "lora_rank": rank,
        "lora_alpha": alpha
    }
})
```

#### Validation Rules

- Both must be `int` (not float)
- Both must be > 0
- No validation on ratio (user decides alpha/rank relationship)

#### Error Messages

```python
builder.with_lora_params(16.5, 32)
# TypeError: rank must be int, got float

builder.with_lora_params(0, 16)
# ValueError: rank must be positive, got 0

builder.with_lora_params(-8, 16)
# ValueError: rank must be positive, got -8
```

#### Implementation

```python
def with_lora_params(self, rank: int, alpha: int) -> 'TorchtuneConfigBuilder':
    """Set LoRA rank and alpha together."""

    # Type validation
    if not isinstance(rank, int):
        raise TypeError(f"rank must be int, got {type(rank).__name__}")
    if not isinstance(alpha, int):
        raise TypeError(f"alpha must be int, got {type(alpha).__name__}")

    # Value validation
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    return self.override({
        "model": {
            "lora_rank": rank,
            "lora_alpha": alpha
        }
    })
```

#### Test Requirements

```python
# Unit tests
def test_with_lora_params_valid()
def test_with_lora_params_preserves_other_model_fields()
def test_with_lora_params_type_errors()
def test_with_lora_params_value_errors()
def test_with_lora_params_chaining()
def test_with_lora_params_custom_ratio()

# Integration test
def test_with_lora_params_validates()
```

---

### Method 2: with_gradient_accumulation_steps()

**Category**: Scientific parameter (affects effective batch size and training dynamics)

#### Signature

```python
def with_gradient_accumulation_steps(self, steps: int) -> 'TorchtuneConfigBuilder':
    """
    Set gradient accumulation steps.

    Gradient accumulation allows training with larger effective batch sizes by
    accumulating gradients over multiple forward passes before updating weights.

    Effective batch size = batch_size * gradient_accumulation_steps

    This affects training dynamics and convergence, making it a scientific parameter.

    Args:
        steps: Number of gradient accumulation steps (positive integer, typically 1-32)
               steps=1 means no accumulation (update after every batch)

    Returns:
        Self for method chaining

    Raises:
        TypeError: If steps is not an int
        ValueError: If steps <= 0

    Example:
        >>> # Increase effective batch size without increasing memory
        >>> builder.with_batch_size(2)
        >>> builder.with_gradient_accumulation_steps(8)  # Effective batch size = 16

        >>> # Sweep accumulation steps
        >>> for steps in [1, 2, 4, 8]:
        ...     builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
        ...     builder.with_gradient_accumulation_steps(steps)
        ...     builder.save(f"configs/grad_accum_{steps}.yaml")
    """
```

#### Behavior

**Config path modified**: `gradient_accumulation_steps`

**Override call**:
```python
return self.override({"gradient_accumulation_steps": steps})
```

#### Validation Rules

- Must be `int`
- Must be > 0
- No upper bound (though typically 1-32)

#### Implementation

```python
def with_gradient_accumulation_steps(self, steps: int) -> 'TorchtuneConfigBuilder':
    """Set gradient accumulation steps."""

    if not isinstance(steps, int):
        raise TypeError(f"steps must be int, got {type(steps).__name__}")

    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    return self.override({"gradient_accumulation_steps": steps})
```

---

### Method 3: with_max_seq_length()

**Category**: Scientific parameter (affects model capacity and training data)

#### Signature

```python
def with_max_seq_length(self, length: int) -> 'TorchtuneConfigBuilder':
    """
    Set maximum sequence length for tokenizer.

    Controls the maximum number of tokens in training sequences. Longer sequences
    capture more context but require more memory.

    Args:
        length: Maximum sequence length (positive integer, typically 512-8192)
                Common values: 512, 1024, 2048, 4096, 8192

    Returns:
        Self for method chaining

    Raises:
        TypeError: If length is not an int
        ValueError: If length <= 0

    Example:
        >>> builder.with_max_seq_length(2048)

        >>> # For long-context tasks
        >>> builder.with_max_seq_length(8192)

        >>> # Note: max_seq_len in config can be null (no limit)
        >>> # This method sets an explicit limit
    """
```

#### Behavior

**Config path modified**: `tokenizer.max_seq_len`

**Override call**:
```python
return self.override({"tokenizer": {"max_seq_len": length}})
```

**Note**: Config default is often `null` (no limit). This method sets an explicit limit.

#### Validation Rules

- Must be `int`
- Must be > 0
- No upper bound validation (model-dependent)

#### Implementation

```python
def with_max_seq_length(self, length: int) -> 'TorchtuneConfigBuilder':
    """Set maximum sequence length."""

    if not isinstance(length, int):
        raise TypeError(f"length must be int, got {type(length).__name__}")

    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    return self.override({"tokenizer": {"max_seq_len": length}})
```

---

### Method 4: with_activation_checkpointing()

**Category**: Engineering parameter (affects memory, not scientific results)

#### Signature

```python
def with_activation_checkpointing(self, enabled: bool) -> 'TorchtuneConfigBuilder':
    """
    Enable or disable activation checkpointing.

    Activation checkpointing trades compute for memory by recomputing activations
    during backward pass instead of storing them. Reduces memory usage but increases
    training time (~20-30% slower).

    This is an engineering parameter - it affects resource usage but not the
    final model's behavior or accuracy.

    Args:
        enabled: True to enable checkpointing (lower memory, slower)
                 False to disable (higher memory, faster)

    Returns:
        Self for method chaining

    Raises:
        TypeError: If enabled is not a bool

    Example:
        >>> # Enable for large models that don't fit in memory
        >>> builder.with_activation_checkpointing(True)

        >>> # Disable for faster training when memory allows
        >>> builder.with_activation_checkpointing(False)

        >>> # Common pattern: enable for training, disable for evaluation
        >>> train_builder.with_activation_checkpointing(True)
        >>> eval_builder.with_activation_checkpointing(False)
    """
```

#### Behavior

**Config path modified**: `enable_activation_checkpointing`

**Override call**:
```python
return self.override({"enable_activation_checkpointing": enabled})
```

#### Validation Rules

- Must be `bool` (not int, str, etc.)

#### Implementation

```python
def with_activation_checkpointing(self, enabled: bool) -> 'TorchtuneConfigBuilder':
    """Enable or disable activation checkpointing."""

    if not isinstance(enabled, bool):
        raise TypeError(f"enabled must be bool, got {type(enabled).__name__}")

    return self.override({"enable_activation_checkpointing": enabled})
```

---

### Method 5: with_seed()

**Category**: Scientific parameter (affects reproducibility)

#### Signature

```python
def with_seed(self, seed: int) -> 'TorchtuneConfigBuilder':
    """
    Set random seed for reproducibility.

    Controls random number generation for initialization, data shuffling, and
    dropout. Same seed + same config = same results (deterministic training).

    Setting seed is critical for scientific reproducibility.

    Args:
        seed: Random seed (non-negative integer)
              Use None to allow non-deterministic behavior (set via override())

    Returns:
        Self for method chaining

    Raises:
        TypeError: If seed is not an int
        ValueError: If seed < 0

    Example:
        >>> # For reproducible experiments
        >>> builder.with_seed(42)

        >>> # Different seeds for multiple runs
        >>> for seed in [42, 123, 456]:
        ...     builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
        ...     builder.with_seed(seed)
        ...     builder.save(f"configs/seed_{seed}.yaml")

        >>> # To remove seed (non-deterministic), use override:
        >>> builder.override({"seed": None})
    """
```

#### Behavior

**Config path modified**: `seed`

**Override call**:
```python
return self.override({"seed": seed})
```

**Note**: Config default is often `null`. This method sets an explicit seed.

#### Validation Rules

- Must be `int`
- Must be >= 0 (non-negative)

#### Implementation

```python
def with_seed(self, seed: int) -> 'TorchtuneConfigBuilder':
    """Set random seed."""

    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    return self.override({"seed": seed})
```

---

### Method 6: with_dtype()

**Category**: Engineering parameter (affects precision and memory)

#### Signature

```python
def with_dtype(self, dtype: str) -> 'TorchtuneConfigBuilder':
    """
    Set training dtype (precision).

    Controls numerical precision for training. Lower precision uses less memory
    but may affect numerical stability. This is an engineering parameter - it
    affects resource usage and training speed, not the scientific approach.

    Common dtypes:
    - "bf16" (bfloat16): Good balance, widely supported, default for most models
    - "fp16" (float16): Smaller memory, but narrower range (can underflow)
    - "fp32" (float32): Full precision, highest memory, most stable

    Args:
        dtype: Data type string, typically "bf16", "fp16", or "fp32"

    Returns:
        Self for method chaining

    Raises:
        TypeError: If dtype is not a str
        ValueError: If dtype is empty

    Example:
        >>> # Default: bfloat16 (good for most cases)
        >>> builder.with_dtype("bf16")

        >>> # Full precision for debugging numerical issues
        >>> builder.with_dtype("fp32")

        >>> # Half precision for maximum memory savings
        >>> builder.with_dtype("fp16")

        >>> # Note: GPU must support the dtype (A100+ for bf16)
    """
```

#### Behavior

**Config path modified**: `dtype`

**Override call**:
```python
return self.override({"dtype": dtype})
```

#### Validation Rules

- Must be `str`
- Must not be empty
- We don't validate specific values (bf16, fp16, fp32) - let torchtune catch invalid dtypes

#### Implementation

```python
def with_dtype(self, dtype: str) -> 'TorchtuneConfigBuilder':
    """Set training dtype."""

    if not isinstance(dtype, str):
        raise TypeError(f"dtype must be str, got {type(dtype).__name__}")

    if not dtype:
        raise ValueError("dtype cannot be empty")

    return self.override({"dtype": dtype})
```

---

### Method 7: with_packed()

**Category**: Engineering parameter (affects GPU efficiency and speed)

#### Signature

```python
def with_packed(self, enabled: bool) -> 'TorchtuneConfigBuilder':
    """
    Enable or disable sequence packing in dataset.

    Packing combines multiple short sequences into a single training example
    to improve GPU utilization. This increases training speed by reducing
    padding overhead, especially for datasets with variable-length sequences.

    This is an engineering parameter - it affects speed and efficiency but
    doesn't fundamentally change what the model learns.

    Args:
        enabled: True to enable packing (faster, better GPU utilization)
                 False to disable packing (simpler, one sequence per example)

    Returns:
        Self for method chaining

    Raises:
        TypeError: If enabled is not a bool

    Example:
        >>> # Enable packing for faster training
        >>> builder.with_packed(True)

        >>> # Disable for simpler training (default in many configs)
        >>> builder.with_packed(False)

        >>> # Common pattern: enable packing for datasets with short sequences
        >>> builder.with_dataset("data/short_instructions.json")
        >>> builder.with_packed(True)  # Pack multiple short sequences together

        >>> # Note: Packing works best when sequences vary in length
    """
```

#### Behavior

**Config path modified**: `dataset.packed`

**Override call**:
```python
return self.override({"dataset": {"packed": enabled}})
```

**Note**: Packing requires the dataset component to support it. Most torchtune datasets support packing, but check the dataset documentation.

#### Validation Rules

- Must be `bool` (not int, str, etc.)

#### Error Messages

```python
builder.with_packed(1)
# TypeError: enabled must be bool, got int

builder.with_packed("yes")
# TypeError: enabled must be bool, got str
```

#### Implementation

```python
def with_packed(self, enabled: bool) -> 'TorchtuneConfigBuilder':
    """Enable or disable dataset packing."""

    if not isinstance(enabled, bool):
        raise TypeError(f"enabled must be bool, got {type(enabled).__name__}")

    return self.override({"dataset": {"packed": enabled}})
```

#### Test Requirements

```python
# Unit tests
def test_with_packed_valid()
def test_with_packed_preserves_other_dataset_fields()
def test_with_packed_type_error()
def test_with_packed_chaining()

# Integration test
def test_with_packed_validates()
```

#### Documentation Notes

**When to use**:
- Enable for datasets with many short sequences (better GPU utilization)
- Enable to speed up training (reduces padding overhead)
- Disable for simpler debugging or when sequence lengths are similar

**How it works**:
- Packs multiple sequences into a single batch element
- Reduces wasted computation on padding tokens
- Can significantly speed up training on datasets with variable lengths

**Trade-offs**:
- **Pros**: Faster training, better GPU utilization
- **Cons**: Slightly more complex data loading, harder to debug individual sequences

**Category**: Engineering parameter (GPU efficiency)

---

### Method 8: with_compile()

**Category**: Engineering parameter (speed optimization)

#### Signature

```python
def with_compile(self, enabled: bool) -> 'TorchtuneConfigBuilder':
    """
    Enable or disable torch.compile for model optimization.

    torch.compile uses TorchDynamo and TorchInductor to compile the model
    and loss function for faster execution. Can significantly speed up training
    but adds compilation overhead at startup.

    Args:
        enabled: True to compile model+loss, False to use eager mode

    Returns:
        Self for method chaining

    Raises:
        TypeError: If enabled is not a bool

    Example:
        >>> # Enable for production training (faster after warmup)
        >>> builder.with_compile(True)

        >>> # Disable for debugging or quick experiments
        >>> builder.with_compile(False)

        >>> # Common pattern: disable for quick tests, enable for full runs
        >>> if args.quick_test:
        ...     builder.with_compile(False)
        ... else:
        ...     builder.with_compile(True)
    """
```

#### Behavior

**Config path modified**: `compile`

**Override call**:
```python
return self.override({"compile": enabled})
```

**Note**: According to torchtune config comment on line 78: "torch.compile the model + loss, True increases speed + decreases memory". First run will be slower due to compilation overhead.

#### Validation Rules

- Must be `bool` (not int, str, etc.)

#### Error Messages

```python
builder.with_compile(1)
# TypeError: enabled must be bool, got int

builder.with_compile("true")
# TypeError: enabled must be bool, got str
```

#### Implementation

```python
def with_compile(self, enabled: bool) -> 'TorchtuneConfigBuilder':
    """Enable or disable torch.compile optimization."""

    if not isinstance(enabled, bool):
        raise TypeError(f"enabled must be bool, got {type(enabled).__name__}")

    return self.override({"compile": enabled})
```

#### Test Requirements

```python
# Unit tests
def test_with_compile_valid()
def test_with_compile_preserves_other_fields()
def test_with_compile_type_error()
def test_with_compile_chaining()

# Integration test
def test_with_compile_validates()
```

#### Documentation Notes

**When to use**:
- Enable for production training runs (significant speedup after warmup)
- Enable when training time is critical
- Disable for debugging (easier error messages)
- Disable for quick experiments (avoid compilation overhead)

**How it works**:
- Uses PyTorch 2.0+ compilation stack (TorchDynamo + TorchInductor)
- Compiles both model and loss function
- First training step is slower (compilation overhead)
- Subsequent steps are faster (optimized kernels)

**Trade-offs**:
- **Pros**: Faster training (after warmup), lower memory usage
- **Cons**: Compilation overhead at startup, harder to debug, requires PyTorch 2.0+

**Performance impact** (from torchtune config comments):
- Speed: Increases (after compilation overhead)
- Memory: Decreases (kernel fusion reduces intermediate activations)

**Category**: Engineering parameter (speed/memory optimization)

---

### Method 9: with_dataset_template()

**Category**: Scientific parameter (affects prompt formatting and learning)

#### Signature

```python
def with_dataset_template(self, template: str) -> 'TorchtuneConfigBuilder':
    """
    Set the prompt template for instruction datasets.

    Controls how instructions/prompts are formatted during training. Different
    templates structure the conversation differently, affecting what the model
    learns. Critical for instruction-following fine-tuning.

    This is a scientific parameter - the template format affects model behavior.

    Common templates:
    - "torchtune.data.AlpacaInstructTemplate" - Alpaca-style instructions
    - "torchtune.data.SummarizeTemplate" - Summarization tasks
    - "torchtune.data.ChatMLTemplate" - ChatML format
    - Custom: "my_module.CustomTemplate" - User-defined templates

    Args:
        template: Full import path to template class
                 Format: "module.path.TemplateClassName"

    Returns:
        Self for method chaining

    Raises:
        TypeError: If template is not a str
        ValueError: If template is empty

    Example:
        >>> # Use Alpaca template
        >>> builder.with_dataset_template("torchtune.data.AlpacaInstructTemplate")

        >>> # Use custom template
        >>> builder.with_dataset_template("my_templates.CustomPromptTemplate")

        >>> # Common pattern: set dataset source and template together
        >>> builder.with_dataset("data/instructions.json")
        >>> builder.with_dataset_template("torchtune.data.AlpacaInstructTemplate")

        >>> # Template affects training behavior
        >>> # Wrong template = model learns wrong format!
    """
```

#### Behavior

**Config path modified**: `dataset.template`

**Override call**:
```python
return self.override({"dataset": {"template": template}})
```

**Note**: The dataset component must support templates (e.g., `torchtune.datasets.instruct_dataset`). Not all datasets use templates.

#### Validation Rules

- Must be `str`
- Must not be empty
- We don't validate the import path exists - let torchtune catch invalid templates at runtime

#### Error Messages

```python
builder.with_dataset_template(123)
# TypeError: template must be str, got int

builder.with_dataset_template("")
# ValueError: template cannot be empty
```

#### Implementation

```python
def with_dataset_template(self, template: str) -> 'TorchtuneConfigBuilder':
    """Set dataset prompt template."""

    if not isinstance(template, str):
        raise TypeError(f"template must be str, got {type(template).__name__}")

    if not template:
        raise ValueError("template cannot be empty")

    return self.override({"dataset": {"template": template}})
```

#### Test Requirements

```python
# Unit tests
def test_with_dataset_template_valid()
def test_with_dataset_template_preserves_other_dataset_fields()
def test_with_dataset_template_type_error()
def test_with_dataset_template_value_error()
def test_with_dataset_template_chaining()

# Integration test
def test_with_dataset_template_validates()
```

#### Documentation Notes

**When to use**:
- Instruction-following fine-tuning
- Changing prompt format for different task types
- Using custom prompt templates

**Important**:
- Template must match your dataset format
- Wrong template = model learns wrong conversation structure
- Templates are Python classes that format the prompts

**Related dataset config**:
```yaml
dataset:
  _component_: torchtune.datasets.instruct_dataset
  source: my/dataset
  template: torchtune.data.AlpacaInstructTemplate  # This field
  train_on_input: False
  column_map:
    output: response
```

**Category**: Scientific parameter (affects what the model learns)

---

## Summary: Complete Method List

**Phase 1 (Essential - 9 methods):**

These methods cover supervised fine-tuning workflows and should be implemented first:

1. `with_dataset(path: str)` - Dataset source (Scientific)
2. `with_output_dir(path: str)` - Output location (Engineering)
3. `with_learning_rate(lr: float)` - Optimizer LR (Scientific)
4. `with_epochs(epochs: int)` - Training duration (Scientific)
5. `with_lora_params(rank: int, alpha: int)` - LoRA architecture (Scientific, paired)
6. `with_batch_size(batch_size: int)` - GPU memory control (Engineering)
7. `with_packed(enabled: bool)` - Sequence packing for efficiency (Engineering)
8. `with_dataset_template(template: str)` - Prompt template for SFT (Scientific)
9. `with_seed(seed: int)` - Reproducibility (Scientific)

**Phase 2+ (Add based on usage):**

Add these based on actual usage patterns:

- `with_gradient_accumulation_steps(steps: int)` - Effective batch size (Scientific)
- `with_max_seq_length(length: int)` - Sequence capacity (Scientific)
- `with_dtype(dtype: str)` - Precision/memory (Engineering)
- `with_activation_checkpointing(enabled: bool)` - Memory/speed tradeoff (Engineering)
- `with_compile(enabled: bool)` - torch.compile optimization (Engineering)

**Rationale**: Phase 1 methods cover supervised fine-tuning (primary use case). Users can always use `override()` for Phase 2+ parameters. Add more based on feedback.

---

## Testing Strategy

**For each method, require:**

1. **Valid input test** - Method works with typical values
2. **Preservation test** - Doesn't modify unrelated config fields
3. **Type error test** - Rejects wrong types
4. **Value error test** - Rejects invalid values (if applicable)
5. **Chaining test** - Returns self for method chaining
6. **Integration test** - Generated config passes `tune validate`

**Example test structure** (see Method 1 for complete examples):
```python
# tests/test_builder.py
def test_with_METHOD_valid()
def test_with_METHOD_preserves_fields()
def test_with_METHOD_type_error()
def test_with_METHOD_value_error()  # If applicable
def test_with_METHOD_chaining()

# tests/test_integration.py
def test_with_METHOD_validates()
```

---

## Usage Patterns

### Pattern 1: Scientific Parameter Configuration

```python
# Focus on what affects the experiment
builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
builder.with_dataset("data/my_data.json")
builder.with_dataset_template("torchtune.data.AlpacaInstructTemplate")
builder.with_learning_rate(3e-4)
builder.with_lora_params(32, 64)  # rank=32, alpha=64
builder.with_gradient_accumulation_steps(8)
builder.with_epochs(3)
builder.with_seed(42)
```

### Pattern 2: Engineering Parameter Configuration (Manual GPU Control)

```python
# GPU efficiency settings (manual control)
# Torchtune configs have good defaults - only override if needed
builder.with_batch_size(2)
builder.with_activation_checkpointing(True)
builder.with_output_dir("results/exp_001")

# Phase 2+ methods (use override() for now)
builder.override({"dtype": "bf16"})
builder.override({"packed": True})  # Enable packing for speed
builder.override({"compile": True})  # Enable torch.compile for speed
```

### Pattern 3: Adapting for Different GPU (One-Time Setup)

```python
# Torchtune configs assume A100-80GB
# If using V100-16GB, adapt once then generate sweep

# Create GPU-adapted base config (one time)
builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
builder.with_batch_size(1)  # Reduce for smaller GPU
builder.with_activation_checkpointing(True)  # Enable for memory
builder.save("configs/v100_base.yaml")

# Generate parameter sweep from adapted base
for lr in [1e-4, 3e-4, 5e-4, 1e-3]:
    builder = TorchtuneConfigBuilder.from_file("configs/v100_base.yaml")
    builder.with_learning_rate(lr)
    builder.with_output_dir(f"results/lr_{lr}")
    builder.save(f"configs/lr_{lr}.yaml")
    builder.validate()
```

---

## Implementation Priority

**Phase 1 (Essential Methods - MVP):**
1. ✅ `with_dataset(path)` - Changes every experiment
2. ✅ `with_output_dir(path)` - Changes every experiment
3. ✅ `with_learning_rate(lr)` - Most tuned hyperparameter
4. ✅ `with_epochs(epochs)` - Common training parameter
5. ✅ `with_lora_params(rank, alpha)` - Paired LoRA parameters
6. ✅ `with_batch_size(batch_size)` - Manual GPU memory control
7. ✅ `with_packed(enabled)` - Sequence packing for variable-length data
8. ✅ `with_dataset_template(template)` - Prompt formatting for SFT (primary use case)
9. ✅ `with_seed(seed)` - Reproducibility for scientific experiments

**Phase 2+ (Add Based on Usage):**
- `with_gradient_accumulation_steps(steps)` - Can use override()
- `with_max_seq_length(length)` - Usually set once per project
- `with_dtype(dtype)` - Usually set once per GPU
- `with_activation_checkpointing(enabled)` - Advanced memory control
- `with_compile(enabled)` - Advanced optimization
- See "Potential Future Methods" section for more

**Decision**: Start with 9 essential methods covering supervised fine-tuning workflows. Users can use `override()` for Phase 2+ parameters. Add more methods based on actual usage patterns and user feedback.

---

## Potential Future Methods

Methods to consider adding based on usage patterns:

**Scientific:**
- `with_weight_decay(wd: float)` - Optimizer regularization
- `with_warmup_steps(steps: int)` - LR scheduler warmup
- `with_lora_dropout(dropout: float)` - LoRA regularization
- `with_clip_grad_norm(max_norm: float)` - Gradient clipping

**Engineering:**
- `with_checkpoint_dir(path: str)` - Model checkpoint location
- `with_device(device: str)` - cuda, cpu, mps
- `with_activation_offloading(enabled: bool)` - Advanced memory management

**Decision**: Add methods when usage patterns emerge, not proactively.

---

## Design Rationale

### Why 6-7 Essential Methods for Phase 1?

1. **High frequency** - These 6-7 methods cover 80% of common modifications
2. **Clear categories** - Mix of scientific (what to learn) and engineering (GPU control)
3. **Meaningful pairing** - Force lora_rank + lora_alpha together (scientifically related)
4. **Escape hatch** - Users can always use `override()` for Phase 2+ parameters
5. **Not over-engineered** - Start simple, add more based on actual usage patterns
6. **SLURM-focused** - Priorities match parameter sweep use case (LR, LoRA, dataset)
7. **Practical** - Torchtune configs have good GPU defaults, manual control when needed

### Why Pair lora_rank + lora_alpha?

- Scientifically related parameters
- Should be considered together
- Prevents incomplete configs
- Makes relationship explicit in code

### Why No Defaults for lora_alpha?

- Convention (alpha=2*rank) is visible in code: `with_lora_params(32, 32*2)`
- Users can choose custom ratios without surprise
- Explicit is better than implicit (Python principle)

---

**Status**: ✅ Complete - Ready for implementation
