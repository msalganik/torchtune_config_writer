# Appendix D: Metadata Format and Tracking

**Purpose**: This appendix defines the metadata format for generated configs, including environment information, data provenance, and reproducibility tracking.

**Audience**: Implementers and anyone needing to understand metadata structure and usage.

**Related sections**: See Appendix B for config loading and `from_previous()` implementation.

**Status**: ✅ Complete

---

## Table of Contents

1. [Overview](#overview)
2. [Design Decisions](#design-decisions)
3. [Metadata Format Specification](#metadata-format-specification)
4. [API Specification](#api-specification)
5. [Privacy and Security](#privacy-and-security)
6. [File Format](#file-format)
7. [Implementation Details](#implementation-details)
8. [Testing Requirements](#testing-requirements)
9. [Phase Roadmap](#phase-roadmap)

---

## Overview

Metadata files (`.meta.yaml`) accompany generated configs to enable:

1. **Reproducibility**: Reconstruct the exact builder state and environment
2. **Provenance**: Track where configs came from and what changed
3. **Auditability**: Know what changed from base config for scientific rigor
4. **Environment tracking**: Capture versions and system info for debugging

**Key principle**: Metadata is separate from config to maintain compatibility with `tune validate` and other tools.

---

## Design Decisions

### 1. Separate Files (Not Embedded)

**Decision**: Store metadata in separate `.meta.yaml` files, not embedded in config

**Rationale**:
- ✅ Clean separation of concerns
- ✅ Config validates with `tune validate` (no extra fields)
- ✅ Easy to share config without metadata (privacy)
- ✅ Metadata can evolve independently
- ❌ Two files to manage (acceptable trade-off)

**Alternative rejected**: Embedding metadata in config YAML
- Would pollute config namespace
- May break `tune validate` if unknown keys rejected
- Harder to use config with other tools
- Privacy concerns (can't share config without metadata)

---

### 2. YAML Format (Not JSON)

**Decision**: Use YAML for metadata files

**Rationale**:
- Consistency with config format
- Human-readable and editable
- PyYAML already a dependency
- Slightly larger/slower than JSON, but not a bottleneck

---

### 3. Phased Metadata Approach

**Decision**: Start simple, enhance over time

**Phase 1 (Simplified metadata)**:
- Source tracking (where config came from)
- Version information (tool, torchtune, Python)
- Override tracking (what changed from base)
- Timestamp
- **No privacy features** (no path sanitization, no dataset hashing, no git info)

**Phase 2 (Enhanced metadata)**:
- Extended environment info (torch, CUDA versions)
- Privacy features (path sanitization, optional dataset hashing)
- Data provenance (dataset paths and hashes - opt-in)
- Optional operation history

**Phase 3 (Advanced features)**:
- Git integration (commit hashes, dirty state - opt-in)
- Custom metadata fields
- Metadata validation and migration

---

### 4. Privacy by Default

**Decision**: Privacy-respecting defaults with opt-in for more info (Phase 2+)

**Phase 1 defaults** (Simplified):
- ✅ Save metadata (reproducibility matters)
- ❌ No path sanitization (Phase 2)
- ❌ No dataset hashing (Phase 2)
- ❌ No hostname (Phase 2)
- ❌ No git info (Phase 2)

**Phase 2+ defaults** (Enhanced):
- ✅ Sanitize absolute paths (replace `/home/username` with `~`)
- ❌ No dataset hashing by default (opt-in for privacy)
- ❌ No hostname by default (infrastructure privacy)
- ❌ No git info by default (may reveal private repos)

**User control** (Phase 2+):
```python
builder.save("config.yaml",
             save_metadata=True,  # Can disable entirely
             sanitize_paths=True,  # Default: True (Phase 2+)
             include_git_info=False,  # Default: False
             hash_dataset=False)  # Default: False
```

---

## Metadata Format Specification

### Phase 1: Simplified Metadata (MVP)

**File**: `config.meta.yaml` (same name as config, with `.meta.yaml` extension)

```yaml
# config.meta.yaml - Phase 1 format (simplified)
metadata_version: "1.0"  # Metadata schema version

# Source information
source_type: "torchtune_shipped"  # or "file", "dict", "previous"
base_config: "llama3_1/8B_lora_single_device"  # if torchtune_shipped
source_file: null  # if from_file: "/path/to/original.yaml"

# Version information
tool_version: "0.2.0"
python_version: "3.11.5"
torchtune_version: "0.6.1"

# Generation info
generated_at: "2025-01-09T14:30:00Z"
generated_by: "torchtune_config_writer"

# Changes from base
overrides:
  optimizer:
    lr: 0.001
  dataset:
    source: "/home/user/data/my_data.json"  # Raw paths (no sanitization in Phase 1)
  output_dir: "/home/user/results/exp_001"
```

**Field descriptions**:

- `metadata_version`: Schema version for forward/backward compatibility
- `source_type`: One of:
  - `"torchtune_shipped"` - From `tune cp` command
  - `"file"` - From user's existing YAML via `from_file()`
  - `"dict"` - From programmatic dict via `from_dict()`
  - `"previous"` - From previously generated config via `from_previous()`
- `base_config`: Torchtune config name (if source_type is "torchtune_shipped")
- `source_file`: Original file path (if source_type is "file" or "previous")
- `tool_version`: Version of torchtune_config_writer
- `python_version`: Python runtime version
- `torchtune_version`: Torchtune library version
- `generated_at`: ISO 8601 timestamp (UTC)
- `generated_by`: Tool identifier
- `overrides`: Nested dict of changes from base config
- `deletions`: List of deleted key paths (dot notation)

---

### Phase 2: Enhanced Metadata

**Additional fields** added in Phase 2:

```yaml
# Extended environment (always included in Phase 2)
environment:
  torch_version: "2.1.0"
  torchao_version: "0.14.0"
  cuda_version: "12.1"  # null if CPU-only
  platform: "Linux-5.15.0-x86_64"

# Path sanitization (enabled by default in Phase 2)
overrides:
  dataset:
    source: "~/data/my_data.json"  # Sanitized path
  output_dir: "~/results/exp_001"

# Data provenance (opt-in: hash_dataset=True)
data_provenance:
  dataset_path: "~/data/my_data.json"  # Sanitized
  dataset_sha256: "abc123..."  # If hash_dataset=True
  dataset_size_bytes: 1048576

# Operation history (opt-in: track_operations=True)
operations:
  - timestamp: "2025-01-09T14:30:00Z"
    type: "override"
    data:
      optimizer:
        lr: 0.001
  - timestamp: "2025-01-09T14:30:05Z"
    type: "delete"
    data: "clip_grad_norm"
  - timestamp: "2025-01-09T14:30:10Z"
    type: "override"
    data:
      dataset:
        source: "~/data/my_data.json"
```

---

### Phase 3: Advanced Metadata

**Additional fields** (opt-in):

```yaml
# Git information (opt-in: include_git_info=True)
git_info:
  commit_hash: "abc123def456"
  branch: "main"
  is_dirty: false
  remote_url: "https://github.com/user/repo"  # Sanitized if needed

# User-defined metadata (opt-in)
custom:
  experiment_name: "baseline"
  researcher: "Alice"
  notes: "First experiment with new dataset"
  tags: ["baseline", "llama3.1", "lora"]
```

---

## API Specification

### Saving with Metadata

```python
class TorchtuneConfigBuilder:
    def save(
        self,
        output_path: str,
        save_metadata: bool = True,
        sanitize_paths: bool = True,
        hash_dataset: bool = False,
        include_git_info: bool = False,
        track_operations: bool = False,
        custom_metadata: Dict[str, Any] = None
    ) -> str:
        """
        Save config and metadata.

        Args:
            output_path: Path for config YAML
            save_metadata: If True, also save .meta.yaml (default: True)
            sanitize_paths: Replace absolute paths with ~ (default: True)
            hash_dataset: Hash dataset file for provenance (default: False, privacy)
            include_git_info: Include git commit info (default: False, privacy)
            track_operations: Save full operation history (default: False)
            custom_metadata: User-defined metadata fields

        Returns:
            Path to saved config

        Raises:
            ValueError: If output_path is invalid
            IOError: If cannot write files
        """
        # Save config
        config = self.build()
        with open(output_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)

        # Save metadata
        if save_metadata:
            metadata = self._generate_metadata(
                sanitize_paths=sanitize_paths,
                hash_dataset=hash_dataset,
                include_git_info=include_git_info,
                track_operations=track_operations,
                custom_metadata=custom_metadata
            )
            meta_path = self._get_metadata_path(output_path)
            with open(meta_path, 'w') as f:
                yaml.dump(metadata, f, sort_keys=False)

        return output_path
```

---

### Metadata Generation (Internal)

```python
def _generate_metadata(
    self,
    sanitize_paths: bool = True,
    hash_dataset: bool = False,
    include_git_info: bool = False,
    track_operations: bool = False,
    custom_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate metadata dict.

    Returns:
        Metadata dictionary ready for YAML serialization
    """
    # Phase 2: Basic metadata
    metadata = {
        "metadata_version": "1.0",
        "source_type": self._source_type,
        "base_config": self._base_config_name if self._source_type == "torchtune_shipped" else None,
        "source_file": self._sanitize_path(self._source_file, sanitize_paths) if self._source_file else None,
        "tool_version": __version__,
        "python_version": platform.python_version(),
        "torchtune_version": self._get_torchtune_version(),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "generated_by": "torchtune_config_writer",
        "overrides": self._compute_overrides(),
        "deletions": self._deletions.copy() if self._deletions else [],
    }

    # Phase 2.5: Extended environment (always include)
    metadata["environment"] = self._capture_environment()

    # Phase 2.5: Data provenance (opt-in)
    if hash_dataset:
        metadata["data_provenance"] = self._capture_data_provenance(sanitize_paths)

    # Phase 2.5: Operation history (opt-in)
    if track_operations:
        metadata["operations"] = self._operations  # Tracked throughout builder lifecycle

    # Phase 3: Git info (opt-in)
    if include_git_info:
        metadata["git_info"] = self._capture_git_info()

    # Phase 3: Custom metadata (opt-in)
    if custom_metadata:
        metadata["custom"] = custom_metadata

    # Remove None values for cleaner YAML
    return {k: v for k, v in metadata.items() if v is not None}
```

---

### Helper Methods

```python
def _get_metadata_path(self, config_path: str) -> str:
    """Get metadata file path from config path."""
    path = Path(config_path)
    return str(path.parent / f"{path.stem}.meta.yaml")

def _sanitize_path(self, path: str, sanitize: bool) -> str:
    """Sanitize path for privacy (replace home with ~)."""
    if not sanitize:
        return path

    # Convert to Path for proper handling
    p = Path(path).expanduser().resolve()
    home = Path.home()

    try:
        # Get relative path from home
        rel = p.relative_to(home)
        return str(Path("~") / rel)
    except ValueError:
        # Not under home directory, return as-is
        return str(p)

def _get_torchtune_version(self) -> str:
    """Get installed torchtune version."""
    import torchtune
    return torchtune.__version__

def _capture_environment(self) -> Dict[str, Any]:
    """Capture extended environment information."""
    import torch
    try:
        import torchao
        torchao_version = torchao.__version__
    except ImportError:
        torchao_version = None

    # CUDA version (if available)
    cuda_version = None
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda

    return {
        "torch_version": torch.__version__,
        "torchao_version": torchao_version,
        "cuda_version": cuda_version,
        "platform": platform.platform(),
    }

def _capture_data_provenance(self, sanitize_paths: bool) -> Dict[str, Any]:
    """Capture dataset provenance information."""
    # Extract dataset path from config
    config = self.build()
    dataset_path = config.get("dataset", {}).get("source")

    if not dataset_path:
        return {}

    # Resolve path
    path = Path(dataset_path).expanduser().resolve()

    if not path.exists():
        return {
            "dataset_path": self._sanitize_path(str(path), sanitize_paths),
            "dataset_exists": False
        }

    # Hash file (expensive for large files!)
    import hashlib
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)

    return {
        "dataset_path": self._sanitize_path(str(path), sanitize_paths),
        "dataset_sha256": sha256.hexdigest(),
        "dataset_size_bytes": path.stat().st_size,
        "dataset_exists": True
    }

def _capture_git_info(self) -> Dict[str, Any]:
    """Capture git repository information."""
    try:
        # Check if in git repo
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            check=True
        )

        # Get commit hash
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()

        # Get branch
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()

        # Check if dirty
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
        is_dirty = len(status) > 0

        # Get remote URL (if exists)
        remote_url = None
        try:
            remote_url = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
        except subprocess.CalledProcessError:
            pass

        return {
            "commit_hash": commit,
            "branch": branch,
            "is_dirty": is_dirty,
            "remote_url": remote_url
        }

    except subprocess.CalledProcessError:
        # Not in a git repo
        return {}

def _compute_overrides(self) -> Dict[str, Any]:
    """Compute override delta from base config."""
    # This is the diff between base config and final config
    # Implementation depends on how we track operations
    # For now, track all override() calls
    return self._overrides  # Accumulated throughout builder lifecycle
```

---

### Loading Metadata

```python
@classmethod
def from_previous(cls, config_path: str, require_metadata: bool = True):
    """
    Load from previously generated config.

    Reads both config.yaml and config.meta.yaml to reconstruct builder.

    Args:
        config_path: Path to config YAML
        require_metadata: Raise error if metadata missing (default: True)

    Returns:
        Reconstructed builder instance

    Raises:
        FileNotFoundError: If config or metadata not found (when required)
        ValueError: If metadata is invalid

    Example:
        >>> builder = TorchtuneConfigBuilder.from_previous("configs/exp_042.yaml")
        >>> builder.with_learning_rate(1e-3)  # Modify
        >>> builder.save("configs/exp_043.yaml")
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load metadata
    meta_path = cls._get_metadata_path(config_path)
    if not Path(meta_path).exists():
        if require_metadata:
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        else:
            # Fall back to from_file
            return cls.from_file(config_path)

    with open(meta_path) as f:
        metadata = yaml.safe_load(f)

    # Validate metadata
    if metadata.get("metadata_version") != "1.0":
        raise ValueError(f"Unsupported metadata version: {metadata.get('metadata_version')}")

    # Reconstruct builder based on source_type
    source_type = metadata.get("source_type")

    if source_type == "torchtune_shipped":
        # Start from same base config
        builder = cls(metadata["base_config"])
    elif source_type in ["file", "previous"]:
        # Use the loaded config as base
        builder = cls.from_dict(config)
        builder._source_type = "previous"
        builder._source_file = config_path
    elif source_type == "dict":
        # Use the loaded config as base
        builder = cls.from_dict(config)
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    # Apply tracked overrides (this reconstructs the builder state)
    if "overrides" in metadata:
        builder.override(metadata["overrides"])

    # Apply deletions
    if "deletions" in metadata:
        for path in metadata["deletions"]:
            builder.delete(path)

    # Store metadata for reference
    builder._loaded_metadata = metadata

    return builder
```

---

## Privacy and Security

### Path Sanitization

**Issue**: Paths may contain sensitive information
- `/home/alice/secret-project/data.json`
- `/Users/bob/personal/experiments/config.yaml`

**Solution**: Sanitize by default
```python
# Original path
"/home/alice/data/dataset.json"

# Sanitized (relative to home)
"~/data/dataset.json"

# Non-home paths kept as absolute
"/mnt/shared/data/dataset.json"  # Outside home, keep as-is
```

**Implementation**: `_sanitize_path()` method (see above)

---

### Dataset Hashing Privacy

**Issue**: Hashing datasets reveals content to anyone with the dataset

**Decision**: Opt-in only (default: False)

**Use cases for hashing**:
- Public datasets (already known)
- Team datasets (internal sharing)
- Reproducibility verification (same dataset?)

**Use cases against hashing**:
- Private/proprietary datasets
- Personal datasets
- Datasets with PII

---

### Hostname Privacy

**Issue**: Hostnames may reveal infrastructure
- `gpu-node-01.company.internal`
- `alice-workstation.local`

**Decision**: Don't include hostname by default

**Future**: Add opt-in `include_hostname` parameter if users want it for multi-machine tracking

---

### Git Repository Privacy

**Issue**: Git info may reveal private repositories or unreleased work

**Decision**: Opt-in only (default: False)

**Use cases for git info**:
- Open source projects
- Internal team tracking
- Debugging version issues

**Use cases against git info**:
- Private repositories
- Unreleased research
- Sensitive code

---

### Metadata Sharing Best Practices

**Documentation to include**:

```markdown
## Sharing Configs and Metadata

**Sharing configs publicly** (e.g., in papers, GitHub):
- ✅ Safe: Share config YAML alone (no metadata)
- ✅ Safe: Share metadata with `sanitize_paths=True` and `hash_dataset=False`
- ⚠️  Careful: Sharing metadata may reveal paths, dataset sizes, environment
- ❌ Don't: Share metadata with absolute paths to private data

**Sharing within team**:
- ✅ Share both config and metadata for reproducibility
- ✅ Use `hash_dataset=True` to verify everyone has same data
- ✅ Use `include_git_info=True` to track code versions

**Example**:
```python
# For public sharing
builder.save("config.yaml",
             sanitize_paths=True,  # Remove username from paths
             hash_dataset=False)   # Don't reveal dataset info

# For team sharing
builder.save("config.yaml",
             sanitize_paths=False,  # Keep full paths
             hash_dataset=True,     # Verify dataset versions
             include_git_info=True) # Track code version
```
```

---

## File Format

### Naming Convention

```
configs/
├── exp_001.yaml       # Config file
├── exp_001.meta.yaml  # Metadata file
├── exp_002.yaml
└── exp_002.meta.yaml
```

**Pattern**: `{config_name}.meta.yaml`

**Rationale**:
- Clear association between config and metadata
- Easy to find corresponding metadata
- Standard pattern (`.meta`, `.metadata` commonly used)

---

### Example Files

**exp_001.yaml** (config):
```yaml
# Torchtune config (unchanged, validates with tune validate)
model:
  _component_: torchtune.models.llama3_1.lora_llama3_1_8b
  lora_rank: 32
  lora_alpha: 64

dataset:
  _component_: torchtune.datasets.alpaca_dataset
  source: /home/alice/data/my_data.json

optimizer:
  _component_: torch.optim.AdamW
  lr: 0.0003

output_dir: /home/alice/results/exp_001
```

**exp_001.meta.yaml** (metadata):
```yaml
metadata_version: '1.0'
source_type: torchtune_shipped
base_config: llama3_1/8B_lora_single_device
source_file: null
tool_version: 0.1.0
python_version: 3.11.5
torchtune_version: 0.6.1
generated_at: '2025-01-09T14:30:00Z'
generated_by: torchtune_config_writer

overrides:
  model:
    lora_rank: 32
    lora_alpha: 64
  dataset:
    source: ~/data/my_data.json
  optimizer:
    lr: 0.0003
  output_dir: ~/results/exp_001

deletions: []

environment:
  torch_version: 2.1.0
  torchao_version: 0.14.0
  cuda_version: '12.1'
  platform: Linux-5.15.0-x86_64
```

---

## Implementation Details

### Tracking Operations (Optional, Phase 2.5)

If `track_operations=True`, track all operations in order:

```python
class TorchtuneConfigBuilder:
    def __init__(self, base_config: str):
        # ... existing init code ...
        self._operations = []  # Track operations

    def override(self, updates: Dict[str, Any]) -> Self:
        # ... existing override code ...

        # Track operation (if enabled)
        self._operations.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "override",
            "data": updates
        })

        return self

    def delete(self, path: str) -> Self:
        # ... existing delete code ...

        # Track operation (if enabled)
        self._operations.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "delete",
            "data": path
        })

        return self
```

**Trade-off**: More detailed history but larger metadata files. Optional feature.

---

### Version Migration (Future)

When metadata schema changes:

```python
def _migrate_metadata(self, metadata: Dict, from_version: str, to_version: str) -> Dict:
    """Migrate metadata between schema versions."""
    if from_version == "1.0" and to_version == "1.1":
        # Add new fields with defaults
        metadata["new_field"] = "default_value"

    return metadata
```

---

## Testing Requirements

### Metadata Generation Tests

```python
def test_basic_metadata_generation():
    """Test Phase 2 basic metadata."""
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder.with_dataset("data/test.json")
    builder.with_learning_rate(1e-3)
    builder.save("test.yaml")

    # Load metadata
    with open("test.meta.yaml") as f:
        meta = yaml.safe_load(f)

    assert meta["metadata_version"] == "1.0"
    assert meta["source_type"] == "torchtune_shipped"
    assert meta["base_config"] == "llama3_1/8B_lora_single_device"
    assert "generated_at" in meta
    assert "overrides" in meta
    assert meta["overrides"]["optimizer"]["lr"] == 1e-3

def test_path_sanitization():
    """Test path sanitization for privacy."""
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder.with_dataset("/home/alice/data/test.json")
    builder.save("test.yaml", sanitize_paths=True)

    with open("test.meta.yaml") as f:
        meta = yaml.safe_load(f)

    # Path should be sanitized
    assert meta["overrides"]["dataset"]["source"] == "~/data/test.json"

def test_metadata_round_trip():
    """Test from_previous() reconstructs correctly."""
    # Create and save
    builder1 = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder1.with_learning_rate(1e-3)
    builder1.with_dataset("data/test.json")
    builder1.save("test.yaml")

    # Load and verify
    builder2 = TorchtuneConfigBuilder.from_previous("test.yaml")
    config1 = builder1.build()
    config2 = builder2.build()

    assert config1 == config2

def test_metadata_optional():
    """Test save without metadata."""
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder.save("test.yaml", save_metadata=False)

    assert Path("test.yaml").exists()
    assert not Path("test.meta.yaml").exists()

def test_missing_metadata_handling():
    """Test from_previous() without metadata."""
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder.save("test.yaml", save_metadata=False)

    # Should raise error by default
    with pytest.raises(FileNotFoundError):
        TorchtuneConfigBuilder.from_previous("test.yaml")

    # Should fall back to from_file if allowed
    builder2 = TorchtuneConfigBuilder.from_previous("test.yaml", require_metadata=False)
    assert builder2 is not None
```

---

### Environment Capture Tests

```python
def test_environment_capture():
    """Test extended environment information."""
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder.save("test.yaml")

    with open("test.meta.yaml") as f:
        meta = yaml.safe_load(f)

    assert "environment" in meta
    assert "torch_version" in meta["environment"]
    assert "platform" in meta["environment"]
```

---

### Privacy Tests

```python
def test_no_dataset_hash_by_default():
    """Test dataset not hashed by default."""
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder.with_dataset("data/test.json")
    builder.save("test.yaml")

    with open("test.meta.yaml") as f:
        meta = yaml.safe_load(f)

    assert "data_provenance" not in meta

def test_dataset_hash_opt_in():
    """Test dataset hashing when opted in."""
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder.with_dataset("data/test.json")
    builder.save("test.yaml", hash_dataset=True)

    with open("test.meta.yaml") as f:
        meta = yaml.safe_load(f)

    assert "data_provenance" in meta
    assert "dataset_sha256" in meta["data_provenance"]

def test_git_info_opt_in():
    """Test git info only included when opted in."""
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")

    # Default: no git info
    builder.save("test1.yaml")
    with open("test1.meta.yaml") as f:
        meta1 = yaml.safe_load(f)
    assert "git_info" not in meta1

    # Opt-in: git info included
    builder.save("test2.yaml", include_git_info=True)
    with open("test2.meta.yaml") as f:
        meta2 = yaml.safe_load(f)
    # Only if in git repo
    if Path(".git").exists():
        assert "git_info" in meta2
```

---

## Phase Roadmap

### Phase 1: Simplified Metadata (Current Implementation)

**Scope**:
- ✅ Metadata version tracking
- ✅ Source tracking (type, base config, source file)
- ✅ Version information (tool, Python, torchtune)
- ✅ Timestamp
- ✅ Overrides tracking
- ✅ Save to `.meta.yaml`
- ❌ No path sanitization (Phase 2)
- ❌ No privacy features (Phase 2)
- ❌ No environment tracking (Phase 2)

**Implementation priority**: Minimal metadata for MVP

**Note**: `from_previous()` is Phase 2, so metadata roundtrip not needed in Phase 1

---

### Phase 2: Enhanced Metadata

**Scope**:
- 🚧 Load with `from_previous()` (metadata roundtrip)
- 🚧 Deletion tracking for `delete()` method
- 🚧 Extended environment (torch, CUDA, platform)
- 🚧 Path sanitization (privacy by default)
- 🚧 Data provenance (opt-in hashing)
- 🚧 Operation history (opt-in tracking)

**Implementation priority**: Required for Phase 2 deliverables

---

### Phase 3: Advanced Features

**Scope**:
- 🚧 Git integration (commit, branch, dirty state - opt-in)
- 🚧 Custom metadata fields
- 🚧 Metadata validation
- 🚧 Schema migration
- 🚧 Metadata diff tool

**Implementation priority**: Post-MVP enhancements

---

## Alignment with Guiding Principles

**Scientific**:
- ✅ Reproducibility through detailed metadata
- ✅ Auditability through change tracking
- ✅ Provenance for scientific rigor

**Modular**:
- ✅ Separate metadata files (clean separation)
- ✅ Optional features (pay for what you use)
- ✅ Extensible (custom metadata, new fields)

**Practical**:
- ✅ Start simple (Phase 2), enhance over time
- ✅ Privacy by default
- ✅ Human-readable YAML

**Privacy Respecting**:
- ✅ Sanitize paths by default
- ✅ No dataset hashing by default
- ✅ No hostname/git info by default
- ✅ User control over what's captured
- ✅ Easy to share configs without metadata

**Self Improving**:
- ✅ Metadata enables learning from past experiments
- ✅ Track what worked (via `from_previous()`)
- ✅ Foundation for experiment management tools

**Tested**:
- ✅ Comprehensive test coverage
- ✅ Round-trip tests
- ✅ Privacy tests
- ✅ Error handling tests

---

**End of Appendix D**
