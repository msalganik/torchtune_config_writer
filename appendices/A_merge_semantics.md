# Appendix A: Config Merge Semantics

**Purpose**: This appendix defines the precise behavior of the `override()` and `delete()` methods, including merge rules, type compatibility, edge cases, and implementation details.

**Audience**: Implementers and anyone needing to understand the exact merge behavior.

**Related sections**: See main SPEC.md for overview and architecture.

---

## Config Merge Semantics

The `override()` method performs deep merging of configuration dictionaries. This section defines precise, predictable merge behavior.

### Core Principles

1. **Simplicity**: Clear recursive algorithm with type-based behavior
2. **Predictability**: Operations apply in call order
3. **Safety**: Type checking prevents silent errors
4. **Compatibility**: Respects torchtune's use of `None` and numeric types

### The Algorithm

**One recursive merge rule:**
- Iterate through override keys and merge each value based on its type

**Three type-based behaviors:**
1. **Scalars** (str, int, float, bool, None) → Replace
2. **Lists** → Replace entirely
3. **Dicts** → Recurse

**Additional features:**
- **Type safety**: Compatible types only (prevents silent errors)
- **Deletion support**: Explicit `delete()` method
- **Operation ordering**: Apply changes in call order

### Merge Rules by Type

#### 1. Scalars → Replace

**Types**: `str`, `int`, `float`, `bool`, `None`

**Rule**: Override value replaces base value

```python
# Numbers
base = {"lr": 1e-4}
override = {"lr": 3e-4}
result = {"lr": 3e-4}

# None is a valid value (torchtune uses None for optional params)
base = {"seed": 42}
override = {"seed": None}
result = {"seed": None}
```

#### 2. Lists → Replace

**Rule**: Override list completely replaces base list (no extending/appending)

```python
base = {"lora_attn_modules": ["q_proj", "v_proj", "output_proj"]}
override = {"lora_attn_modules": ["k_proj"]}
result = {"lora_attn_modules": ["k_proj"]}  # Fully replaced
```

**Rationale**: ML configs use lists for discrete choices. Extending would accumulate unwanted values.

**To extend explicitly:**
```python
config = builder.build()
modules = config["model"]["lora_attn_modules"]
builder.override({"model": {"lora_attn_modules": modules + ["k_proj"]}})
```

#### 3. Dicts → Deep Merge (Recurse)

**Rule**: Recursively merge, preserving keys not in override

```python
# Simple case
base = {
    "optimizer": {
        "lr": 1e-4,
        "weight_decay": 0.01,
        "fused": True
    }
}
override = {"optimizer": {"lr": 3e-4}}
result = {
    "optimizer": {
        "lr": 3e-4,           # Updated
        "weight_decay": 0.01, # Preserved
        "fused": True         # Preserved
    }
}

# Multi-level nesting
base = {
    "profiler": {
        "enabled": False,
        "trace_options": {"profile_memory": False, "with_stack": False},
        "schedule": {"wait_steps": 5}
    }
}
override = {
    "profiler": {
        "enabled": True,
        "trace_options": {"profile_memory": True}
    }
}
result = {
    "profiler": {
        "enabled": True,
        "trace_options": {"profile_memory": True, "with_stack": False},
        "schedule": {"wait_steps": 5}  # Entirely preserved
    }
}

# Component fields (no special handling - just normal dict merge)
base = {
    "model": {
        "_component_": "torchtune.models.llama3_1.lora_llama3_1_8b",
        "lora_rank": 8,
        "lora_alpha": 16
    }
}
override = {"model": {"lora_rank": 32}}
result = {
    "model": {
        "_component_": "torchtune.models.llama3_1.lora_llama3_1_8b",  # Preserved
        "lora_rank": 32,                                               # Updated
        "lora_alpha": 16                                               # Preserved
    }
}
```

### Type Compatibility

Types must match with these exceptions:

1. **Exact match**: Always allowed
2. **None → any type**: Allowed (torchtune uses `None` for optional params)
3. **int ↔ float**: Allowed (YAML parsers interpret `2` as int or float inconsistently)
4. **Everything else**: Raises `TypeError`

**Examples:**

```python
# Incompatible types
base = {"lr": 3e-4}
override = {"lr": "high"}
# Raises: TypeError("Cannot override float with str at key 'lr'")

base = {"modules": ["q_proj"]}
override = {"modules": "k_proj"}
# Raises: TypeError("Cannot override list with str at key 'modules'")

# Compatible: None → value
base = {"seed": None}
override = {"seed": 42}
result = {"seed": 42}  ✓

# Compatible: int ↔ float
base = {"batch_size": 2}      # int
override = {"batch_size": 2.0}  # float
result = {"batch_size": 2.0}  ✓
```

### Deleting Keys

**Phase**: Phase 2 (deferred from Phase 1 MVP)

Use explicit `delete()` method:

```python
# Top-level key
builder.delete("clip_grad_norm")

# Nested key (dot notation)
builder.delete("optimizer.weight_decay")
builder.delete("profiler.trace_options.with_stack")

# Non-existent key is a no-op (safe)
builder.delete("nonexistent.key")

# Can be chained
builder.delete("clip_grad_norm").delete("max_steps_per_epoch")
```

**Path format rules:**
- Dot-separated: `"key"`, `"parent.child"`, `"a.b.c"`
- No leading/trailing dots: `".key"` ❌
- No double dots: `"a..b"` ❌
- No empty string: `""` ❌

**Rationale**: Explicit deletion is clearer than overloading `None` with dual meaning (torchtune uses `None` as a valid value).

### Operation Ordering

**Critical**: Operations apply in **call order**, not grouped by type.

```python
# Example 1: Multiple overrides accumulate
builder.override({"optimizer": {"lr": 1e-3}})
builder.override({"optimizer": {"weight_decay": 0.02}})
builder.override({"model": {"lora_rank": 16}})
# Result via deep merge:
# - optimizer.lr = 1e-3 (from first)
# - optimizer.weight_decay = 0.02 (from second)
# - optimizer.fused = True (preserved from base)
# - model.lora_rank = 16 (from third)

# Example 2: Delete then override
builder.override({"lr": 1e-4})
builder.delete("lr")
builder.override({"lr": 5e-4})
# Result: lr = 5e-4 (last operation wins)

# Example 3: Override same key
builder.override({"lr": 1e-3})
builder.override({"lr": 5e-4})
# Result: lr = 5e-4 (last write wins)
```

### Input Validation

#### override()

**Requirements:**
- Must receive a `dict`
- Raises `TypeError` for non-dict inputs (list, str, None, int, etc.)

```python
builder.override([1, 2, 3])      # TypeError
builder.override("string")       # TypeError
builder.override(None)           # TypeError
```

#### delete()

**Requirements:**
- Must receive a `str` (dot-separated path)
- Path cannot be empty, have double dots, or leading/trailing dots
- Raises `TypeError` for non-string
- Raises `ValueError` for invalid format

```python
builder.delete(123)              # TypeError
builder.delete("")               # ValueError
builder.delete(".lr")            # ValueError
builder.delete("optimizer..lr")  # ValueError
```

### Key Edge Cases

#### Adding New Keys

```python
base = {"lr": 1e-4}
override = {"lr": 3e-4, "weight_decay": 0.01}  # New key
result = {"lr": 3e-4, "weight_decay": 0.01}

# Nested new keys
base = {"optimizer": {"lr": 1e-4}}
override = {"optimizer": {"lr": 3e-4, "amsgrad": True}}
result = {"optimizer": {"lr": 3e-4, "amsgrad": True}}
```

#### Delete Non-existent Path

```python
builder.delete("nonexistent.key.path")
builder.delete("lr")
builder.delete("lr")  # Second delete is no-op
# All safe, no errors
```

#### Very Deep Nesting

```python
# 100+ levels of nesting should work without RecursionError
base = {"a": {"b": {"c": ... }}}  # 100 levels deep
override = {"a": {"b": {"c": ... }}}
result = deep_merge(base, override)  # Works
```

### Implementation Notes

#### Algorithm Pseudocode

```python
def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base."""
    result = base.copy()  # Shallow copy

    for key, override_value in override.items():
        # New key: add it
        if key not in result:
            result[key] = override_value
            continue

        base_value = result[key]

        # Type compatibility check
        if not types_compatible(base_value, override_value):
            raise TypeError(f"Cannot override {type(base_value).__name__} "
                          f"with {type(override_value).__name__} at key '{key}'")

        # Merge by type
        if isinstance(override_value, dict):
            result[key] = deep_merge(base_value, override_value)  # Recurse
        else:
            result[key] = override_value  # Replace scalars and lists

    return result

def types_compatible(base_value: Any, override_value: Any) -> bool:
    """Check type compatibility."""
    if type(base_value) == type(override_value):
        return True
    if base_value is None:
        return True
    if isinstance(base_value, (int, float)) and isinstance(override_value, (int, float)):
        return True
    return False
```

#### Builder Class Structure

```python
class TorchtuneConfigBuilder:
    def __init__(self, base_config_name: str):
        self._base_config = self._load_base(base_config_name)
        self._operations: List[Tuple[str, Any]] = []  # Track operations in order

    def override(self, updates: Dict[str, Any]) -> 'TorchtuneConfigBuilder':
        """Apply updates. Validates input is dict."""
        if not isinstance(updates, dict):
            raise TypeError(f"override() requires dict, got {type(updates).__name__}")
        self._operations.append(("override", updates))
        return self

    # Phase 2: delete() method
    def delete(self, path: str) -> 'TorchtuneConfigBuilder':
        """Delete key by path. Validates path format. (Phase 2)"""
        if not isinstance(path, str):
            raise TypeError(f"delete() requires str, got {type(path).__name__}")
        if not path or ".." in path or path.startswith(".") or path.endswith("."):
            raise ValueError(f"Invalid path format: {path!r}")
        self._operations.append(("delete", path))
        return self

    def build(self) -> Dict[str, Any]:
        """Build final config by applying operations in order."""
        config = copy.deepcopy(self._base_config)  # Deep copy once

        for op_type, op_data in self._operations:
            if op_type == "override":
                config = deep_merge(config, op_data)
            elif op_type == "delete":
                self._delete_path(config, op_data)  # Phase 2

        return config

    # Phase 2: delete path helper
    def _delete_path(self, config: Dict, path: str) -> None:
        """Delete nested key by dot-separated path. (Phase 2)"""
        parts = path.split('.')
        current = config

        # Navigate to parent
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                return  # Path doesn't exist, no-op
            current = current[part]

        # Delete final key
        current.pop(parts[-1], None)
```

**Key design decisions:**
- Operations tracked in order as `List[Tuple[str, Any]]`
- Deep copy base config once at start of `build()`
- Shallow copy at each merge level (not deep - avoids redundant copying)
- Input validation in `override()` and `delete()`
- Delete is safe no-op if path doesn't exist

### Testing Strategy for Merge Logic

**Core Merge Logic** (8 test categories):
1. Scalar replacement: str, int, float, bool, None
2. List replacement: full replacement, empty list
3. Dict deep merge: simple, nested, multi-level
4. Type compatibility: None→value, int↔float conversions
5. Type errors: float→str, list→str, dict→scalar
6. Adding new keys: top-level, nested
7. Empty inputs: empty override, empty base, empty dict override
8. Deep nesting: 100+ level recursion

**Input Validation** (3 test categories):
1. override() type checking: Reject list, str, None, int inputs
2. delete() type checking: Reject non-string inputs
3. delete() format validation: Reject empty, double dots, leading/trailing dots

**Builder Operations** (6 test categories):
1. Delete top-level: Remove key from root
2. Delete nested: Remove key from nested dict
3. Delete non-existent: No error, no-op
4. Delete duplicate: Safe to delete same path twice
5. Operation ordering: Operations apply in call order, last wins
6. Multiple overrides: Accumulate via deep merge
7. Method chaining: All methods return self

**Integration Tests** (2 scenarios):
1. Real torchtune config: Load actual config, override, verify merge correctness, validate
2. Complex workflow: Multi-step scientific param setting, deletions, dataset override

### FAQ

**Q: Why replace lists instead of extending?**

A: Extending would accumulate values across overrides:

```python
# If we extended:
builder.override({"modules": ["q_proj"]})
builder.override({"modules": ["v_proj"]})
# Result: ["q_proj", "v_proj", "q_proj", "v_proj"]  ← duplicates!
```

Replacement is predictable. To extend, do it explicitly:
```python
modules = builder.build()["model"]["modules"]
builder.override({"model": {"modules": modules + ["new"]}})
```

**Q: Why explicit delete() instead of using None?**

A: Torchtune uses `None` as a meaningful value:

```yaml
seed: null                # Valid: no random seed
max_steps_per_epoch: null # Valid: train full epoch
```

Using `None` for deletion creates ambiguity:

```python
builder.override({"seed": None})  # Sets seed to None ✓
builder.delete("seed")             # Removes seed key ✓
```

Explicit is clearer.

**Q: Why allow int ↔ float conversions?**

A: YAML parsers interpret `2` inconsistently as int or float depending on context. Forbidding conversions causes spurious errors:

```yaml
batch_size: 2    # Might parse as int or float
```

Allowing numeric conversions prevents parser differences from breaking merges, while still catching real type errors (e.g., str→list).

**Q: What about _component_ fields?**

A: No special handling needed. Deep merge preserves them automatically:

```python
# Change parameters: _component_ preserved
builder.override({"model": {"lora_rank": 16}})

# Change component: provide new _component_ with all required params
builder.override({
    "model": {
        "_component_": "new.component",
        # All required params
    }
})
```

Let `tune validate` catch component incompatibilities.
