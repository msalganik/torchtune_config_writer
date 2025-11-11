# Appendix B: Config Loading Strategy

**Purpose**: This appendix defines how configurations are loaded from different sources (torchtune shipped configs, user files, dicts, previous configs) and how source provenance is tracked.

**Audience**: Implementers and anyone needing to understand config loading mechanisms.

**Related sections**: See main SPEC.md for overview and architecture.

---

## Overview

The `TorchtuneConfigBuilder` must support loading configurations from multiple sources to enable flexible workflows. This spec defines how configs are loaded, validated, and tracked.

## Core Requirements

1. **Load from torchtune's shipped configs** - Leverage proven recipes
2. **Load from user's custom YAML files** - Support existing configs
3. **Load from dict** - Enable programmatic generation
4. **Load from previously generated configs** - Support iterative workflows
5. **Track provenance** - Know where the base config came from

---

## Loading Methods

### 1. From Torchtune's Shipped Configs (Primary Constructor)

**Use case**: Start from torchtune's proven recipe configs

```python
builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
builder = TorchtuneConfigBuilder("llama3/8B_full")
builder = TorchtuneConfigBuilder("mistral/7B_lora_single_device")
```

**Implementation**:
```python
def __init__(self, base_config_name: str):
    """
    Initialize from torchtune's shipped config.

    Args:
        base_config_name: Config name as shown in `tune ls`
                         Format: "model_family/variant"
                         Examples: "llama3_1/8B_lora_single_device"

    Raises:
        ValueError: If config_name not found in torchtune
        subprocess.CalledProcessError: If tune cp fails
    """
    self._base_config = self._load_from_torchtune(base_config_name)
    self._base_config_source = base_config_name
    self._source_type = "torchtune_shipped"
    self._operations = []
```

**Loading mechanism**:
```python
def _load_from_torchtune(self, config_name: str) -> Dict[str, Any]:
    """Load config using tune cp command."""
    import subprocess
    import tempfile
    import yaml

    # Use tune cp to get config content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name

    try:
        result = subprocess.run(
            ["tune", "cp", config_name, temp_path],
            capture_output=True,
            text=True,
            check=True
        )

        with open(temp_path) as f:
            config = yaml.safe_load(f)

        return config

    except subprocess.CalledProcessError as e:
        # tune cp failed - config doesn't exist
        raise ValueError(
            f"Config '{config_name}' not found. "
            f"Run 'tune ls' to see available configs.\n"
            f"Error: {e.stderr}"
        )
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
```

**Validation**:
- Config name must exist in `tune ls` output
- Loaded YAML must be valid dict
- Should contain expected torchtune fields (model, tokenizer, etc.)

---

### 2. From User's Custom YAML File

**Use case**: Start from an existing custom config file

```python
builder = TorchtuneConfigBuilder.from_file("/path/to/my_config.yaml")
builder = TorchtuneConfigBuilder.from_file("configs/custom_experiment.yaml")
```

**Implementation**:
```python
@classmethod
def from_file(cls, config_path: str) -> 'TorchtuneConfigBuilder':
    """
    Initialize from a YAML config file.

    Args:
        config_path: Path to YAML config file (absolute or relative to CWD)

    Returns:
        TorchtuneConfigBuilder instance

    Raises:
        FileNotFoundError: If config_path doesn't exist
        yaml.YAMLError: If file is not valid YAML
        ValueError: If loaded content is not a dict
    """
    import yaml
    from pathlib import Path

    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must contain a YAML dict, got {type(config).__name__}"
        )

    # Create instance
    instance = cls.__new__(cls)
    instance._base_config = config
    instance._base_config_source = str(config_path)
    instance._source_type = "file"
    instance._operations = []

    return instance
```

**Path resolution**:
- Absolute paths: Used as-is
- Relative paths: Resolved from current working directory (CWD)
- Use `Path.resolve()` to get absolute path for metadata

**Validation**:
- File must exist
- Must be valid YAML
- Root must be a dict (not list, scalar, etc.)
- No validation of torchtune-specific structure (user may have custom schema)

---

### 3. From Dict

**Use case**: Programmatic config generation, testing

```python
config_dict = {
    "model": {"_component_": "torchtune.models.llama3_1.lora_llama3_1_8b"},
    "optimizer": {"lr": 1e-4}
}
builder = TorchtuneConfigBuilder.from_dict(config_dict)
```

**Implementation**:
```python
@classmethod
def from_dict(cls, config: Dict[str, Any]) -> 'TorchtuneConfigBuilder':
    """
    Initialize from a config dict.

    Args:
        config: Configuration dictionary

    Returns:
        TorchtuneConfigBuilder instance

    Raises:
        TypeError: If config is not a dict
    """
    if not isinstance(config, dict):
        raise TypeError(f"Config must be dict, got {type(config).__name__}")

    import copy

    instance = cls.__new__(cls)
    instance._base_config = copy.deepcopy(config)
    instance._base_config_source = "<dict>"
    instance._source_type = "dict"
    instance._operations = []

    return instance
```

**Validation**:
- Input must be a dict
- Deep copied to prevent mutation
- No validation of structure (user's responsibility)

---

### 4. From Previously Generated Config (with Metadata)

**Use case**: Iterate on successful experiments

```python
# Load config that was previously generated by this tool
builder = TorchtuneConfigBuilder.from_previous("configs/exp_042.yaml")

# Modify it
builder.override({"optimizer": {"lr": 5e-4}})
builder.save("configs/exp_043.yaml")
```

**Implementation**:
```python
@classmethod
def from_previous(cls, config_path: str, require_metadata: bool = True) -> 'TorchtuneConfigBuilder':
    """
    Load from a previously generated config (with metadata).

    Reads both config.yaml and config.meta.yaml to reconstruct
    the base config and previously applied operations.

    Args:
        config_path: Path to config YAML file
        require_metadata: If True, raise error if .meta.yaml missing

    Returns:
        TorchtuneConfigBuilder instance

    Raises:
        FileNotFoundError: If config or required metadata not found
        ValueError: If metadata is invalid
    """
    from pathlib import Path
    import yaml

    config_path = Path(config_path).resolve()
    meta_path = config_path.with_suffix('.meta.yaml')

    # Load config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Try to load metadata
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = yaml.safe_load(f)

        # Reconstruct from base + overrides
        base_config_name = metadata.get('base_config')
        overrides = metadata.get('overrides', {})

        if base_config_name:
            # Original was from torchtune shipped config
            instance = cls(base_config_name)
        else:
            # Original was from file or dict
            instance = cls.from_dict(config)
            instance._base_config_source = metadata.get('source', '<unknown>')
            instance._source_type = metadata.get('source_type', 'unknown')

        # Replay overrides (if we have them)
        if overrides:
            instance.override(overrides)

        return instance

    elif require_metadata:
        raise FileNotFoundError(
            f"Metadata file not found: {meta_path}\n"
            f"Cannot reconstruct builder without metadata. "
            f"Use from_file() if you just want to load the config."
        )

    else:
        # No metadata, just load as regular file
        return cls.from_file(str(config_path))
```

**Metadata format** (saved alongside config):
```yaml
# config.meta.yaml
source_type: "torchtune_shipped"  # or "file", "dict"
base_config: "llama3_1/8B_lora_single_device"  # if from torchtune
source: "/path/to/original.yaml"  # if from file
tool_version: "0.1.0"
torchtune_version: "0.6.1"
generated_at: "2025-01-08T14:30:00Z"
overrides:
  optimizer:
    lr: 0.001
  dataset:
    source: "/path/to/data.json"
```

---

## Discovering Available Configs

### List Torchtune's Shipped Configs

```python
@staticmethod
def list_available() -> List[str]:
    """
    List all available configs from torchtune.

    Returns:
        List of config names (e.g., ["llama3_1/8B_lora_single_device", ...])

    Raises:
        subprocess.CalledProcessError: If tune ls fails
    """
    import subprocess

    result = subprocess.run(
        ["tune", "ls"],
        capture_output=True,
        text=True,
        check=True
    )

    # Parse tune ls output
    # Format:
    # RECIPE                      CONFIG
    # full_finetune_single_device llama3_1/8B_full_single_device
    # lora_finetune_single_device llama3_1/8B_lora_single_device

    configs = []
    lines = result.stdout.strip().split('\n')

    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 2:
            config_name = parts[-1]  # Last column is config name
            if config_name and '/' in config_name:
                configs.append(config_name)

    return sorted(set(configs))
```

**Usage**:
```python
# See what's available
available = TorchtuneConfigBuilder.list_available()
print(f"Available configs: {len(available)}")
for config in available[:5]:
    print(f"  - {config}")

# Use one
builder = TorchtuneConfigBuilder(available[0])
```

---

## Source Tracking

Every builder instance tracks where its base config came from:

```python
class TorchtuneConfigBuilder:
    def __init__(self, base_config_name: str):
        self._base_config: Dict[str, Any]      # The loaded config
        self._base_config_source: str          # Where it came from
        self._source_type: str                 # Type: "torchtune_shipped", "file", "dict"
        self._operations: List[Tuple[str, Any]]  # Operations to apply

    @property
    def source(self) -> str:
        """Get the source of the base config."""
        return self._base_config_source

    @property
    def source_type(self) -> str:
        """Get the type of source."""
        return self._source_type
```

**Source types**:
- `"torchtune_shipped"`: Loaded via `tune cp` from torchtune's recipes
- `"file"`: Loaded from user's YAML file
- `"dict"`: Created from dict
- `"unknown"`: Loaded from config without metadata

**Usage in metadata**:
```python
def save(self, output_path: str, save_metadata: bool = True) -> str:
    """Save config and metadata."""
    # ... save config YAML ...

    if save_metadata:
        metadata = {
            "source_type": self._source_type,
            "source": self._base_config_source,
            "tool_version": __version__,
            "torchtune_version": get_torchtune_version(),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "overrides": self._compute_overrides()  # Delta from base
        }

        # Only include base_config if from torchtune
        if self._source_type == "torchtune_shipped":
            metadata["base_config"] = self._base_config_source

        meta_path = Path(output_path).with_suffix('.meta.yaml')
        with open(meta_path, 'w') as f:
            yaml.dump(metadata, f)
```

---

## Path Resolution Strategy

**For user config files**:
- **Absolute paths**: `/home/user/configs/my_config.yaml` → used as-is
- **Relative paths**: `configs/my_config.yaml` → resolved from CWD
- **Always store resolved absolute path** in metadata for reproducibility

**Implementation**:
```python
from pathlib import Path

# When loading
config_path = Path(config_path).resolve()  # Convert to absolute

# In metadata
metadata["source"] = str(config_path)  # Store absolute path
```

**Rationale**:
- CWD resolution matches user expectations (like `python script.py`)
- Storing absolute paths in metadata ensures reproducibility
- Path objects handle cross-platform differences

---

## Error Handling

### Config Not Found (Torchtune)

```python
try:
    builder = TorchtuneConfigBuilder("typo/8B_lora")
except ValueError as e:
    print(e)
    # Config 'typo/8B_lora' not found.
    # Run 'tune ls' to see available configs.
    # Error: ...
```

### File Not Found

```python
try:
    builder = TorchtuneConfigBuilder.from_file("nonexistent.yaml")
except FileNotFoundError as e:
    print(e)
    # Config file not found: /absolute/path/nonexistent.yaml
```

### Invalid YAML

```python
try:
    builder = TorchtuneConfigBuilder.from_file("bad_syntax.yaml")
except yaml.YAMLError as e:
    print(e)
    # Invalid YAML in /path/to/bad_syntax.yaml: ...
```

### Invalid Type

```python
try:
    builder = TorchtuneConfigBuilder.from_dict([1, 2, 3])
except TypeError as e:
    print(e)
    # Config must be dict, got list
```

---

## Usage Examples

### Example 1: Start from Torchtune, Customize

```python
builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
builder.override({"optimizer": {"lr": 1e-3}})
builder.save("configs/my_experiment.yaml")
```

### Example 2: Start from Custom File

```python
# User has existing config
builder = TorchtuneConfigBuilder.from_file("my_custom_config.yaml")
builder.override({"epochs": 5})
builder.save("configs/my_experiment.yaml")
```

### Example 3: Iterate on Previous Success

```python
# Load previous experiment
builder = TorchtuneConfigBuilder.from_previous("configs/exp_042.yaml")

# Tweak one parameter
builder.override({"optimizer": {"lr": 5e-4}})

# Save as new experiment
builder.save("configs/exp_043.yaml")
```

### Example 4: Programmatic Generation

```python
# Generate from scratch
base = {
    "model": {"_component_": "torchtune.models.llama3_1.lora_llama3_1_8b"},
    "optimizer": {"_component_": "torch.optim.AdamW", "lr": 1e-4}
}
builder = TorchtuneConfigBuilder.from_dict(base)
builder.override({"optimizer": {"lr": 3e-4}})
builder.save("configs/generated.yaml")
```

### Example 5: Explore Available Configs

```python
# See what's available
configs = TorchtuneConfigBuilder.list_available()

# Filter for LoRA configs
lora_configs = [c for c in configs if "lora" in c.lower()]
print(f"LoRA configs: {lora_configs}")

# Use one
builder = TorchtuneConfigBuilder(lora_configs[0])
```

---

## Testing Strategy

### Unit Tests

```python
def test_init_from_torchtune():
    """Test loading from torchtune shipped config."""
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    assert builder.source_type == "torchtune_shipped"
    assert builder.source == "llama3_1/8B_lora_single_device"
    assert "model" in builder._base_config

def test_init_invalid_config_name():
    """Test error for non-existent torchtune config."""
    with pytest.raises(ValueError, match="not found"):
        TorchtuneConfigBuilder("nonexistent/config")

def test_from_file():
    """Test loading from custom YAML file."""
    # Create temp file
    config = {"model": {"_component_": "test"}}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        path = f.name

    try:
        builder = TorchtuneConfigBuilder.from_file(path)
        assert builder.source_type == "file"
        assert Path(builder.source).is_absolute()
        assert builder._base_config == config
    finally:
        os.unlink(path)

def test_from_file_not_found():
    """Test error for non-existent file."""
    with pytest.raises(FileNotFoundError):
        TorchtuneConfigBuilder.from_file("/nonexistent/path.yaml")

def test_from_dict():
    """Test loading from dict."""
    config = {"model": {"_component_": "test"}}
    builder = TorchtuneConfigBuilder.from_dict(config)
    assert builder.source_type == "dict"
    assert builder.source == "<dict>"
    assert builder._base_config == config

    # Should deep copy (not reference)
    config["model"]["new_key"] = "value"
    assert "new_key" not in builder._base_config["model"]

def test_from_dict_invalid_type():
    """Test error for non-dict input."""
    with pytest.raises(TypeError):
        TorchtuneConfigBuilder.from_dict([1, 2, 3])

def test_list_available():
    """Test listing available configs."""
    configs = TorchtuneConfigBuilder.list_available()
    assert isinstance(configs, list)
    assert len(configs) > 0
    assert "llama3_1/8B_lora_single_device" in configs
```

### Integration Tests

```python
def test_from_previous_roundtrip():
    """Test save and reload with metadata."""
    # Create config
    builder1 = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder1.override({"optimizer": {"lr": 1e-3}})

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "test.yaml")
        builder1.save(config_path, save_metadata=True)

        # Reload
        builder2 = TorchtuneConfigBuilder.from_previous(config_path)

        # Should have same source
        assert builder2.source_type == "torchtune_shipped"
        assert builder2.source == "llama3_1/8B_lora_single_device"

        # Should have same overrides applied
        config1 = builder1.build()
        config2 = builder2.build()
        assert config1 == config2

def test_from_previous_no_metadata():
    """Test from_previous with require_metadata=False."""
    # Create config without metadata
    config = {"model": {"_component_": "test"}}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        path = f.name

    try:
        # Should fall back to from_file()
        builder = TorchtuneConfigBuilder.from_previous(path, require_metadata=False)
        assert builder.source_type == "file"

        # Should raise if required
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            TorchtuneConfigBuilder.from_previous(path, require_metadata=True)
    finally:
        os.unlink(path)
```

---

## Summary

**Four loading methods:**
1. `TorchtuneConfigBuilder(name)` - From torchtune's shipped configs
2. `TorchtuneConfigBuilder.from_file(path)` - From user's YAML file
3. `TorchtuneConfigBuilder.from_dict(config)` - From dict
4. `TorchtuneConfigBuilder.from_previous(path)` - From previously generated config

**Key features:**
- Source tracking for reproducibility
- Path resolution from CWD for files
- Metadata for reconstructing builder state
- Discovery via `list_available()`
- Clear error messages

**Design decisions:**
- Use `tune cp` to load torchtune configs (authoritative source)
- Store absolute paths in metadata
- Deep copy dicts to prevent mutation
- Require metadata for `from_previous()` by default
- Track source type for appropriate handling

This enables flexible workflows while maintaining reproducibility.
