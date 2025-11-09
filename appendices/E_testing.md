# Appendix E: Testing Strategy

**Purpose**: This appendix defines the comprehensive testing approach for the config builder, including unit tests, integration tests, and test organization.

**Audience**: Implementers writing tests and anyone needing to understand test coverage.

**Related sections**:
- See Appendix A for merge semantics testing details
- See Appendix B for config loading testing details
- See main SPEC.md for implementation phases

---

## Testing Strategy

### Unit Tests

Test core builder functionality without external dependencies:

1. **Initialization**: Load base configs correctly
2. **Override logic**: Deep merge works correctly
3. **High-level methods**: `with_*` methods apply correct overrides
4. **Build**: Final dict merges overrides correctly
5. **Metadata**: Metadata dict contains all required fields
6. **from_dict**: Reconstruct builder from spec
7. **from_previous**: Load from saved config + metadata

### Integration Tests

Test end-to-end with actual torchtune:

1. **Generate and validate**: Create config, run `tune validate`
2. **Multiple recipes**: Test with different base configs
3. **Complex overrides**: Nested dict modifications
4. **Round-trip**: Save, load with `from_previous`, verify identical
5. **Batch generation**: Generate multiple configs, all validate

### Test Organization

```
tests/
├── test_builder.py          # Unit tests for core logic
├── test_integration.py      # Integration tests with tune validate
├── test_metadata.py         # Metadata generation and loading
├── test_gpu_efficiency.py   # GPU helper tests
└── fixtures/
    └── sample_configs/      # Test configs for validation
```

### Example Tests

```python
def test_override_learning_rate():
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder.with_learning_rate(1e-3)

    config = builder.build()
    assert config["optimizer"]["lr"] == 1e-3

def test_generated_config_validates():
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder.with_dataset("tests/fixtures/sample_data.json")

    config_path = builder.save("/tmp/test_config.yaml")

    # Should not raise
    builder.validate(config_path)
```

## Detailed Test Coverage

### Merge Semantics Tests

See **Appendix A: Config Merge Semantics** for detailed testing strategy covering:
- Core merge logic (8 categories)
- Input validation (3 categories)
- Builder operations (6 categories)
- Integration scenarios (2 categories)

### Config Loading Tests

See **Appendix B: Config Loading Strategy** for detailed testing strategy covering:
- Loading from torchtune shipped configs
- Loading from user YAML files
- Loading from dicts
- Loading from previous configs with metadata
- Error handling for each loading method

### Metadata Tests

**Generation Tests:**
1. Metadata contains all required fields
2. Source tracking correct for each loading method
3. Overrides delta computed correctly
4. Timestamps in ISO format
5. Version info captured

**Loading Tests:**
1. from_previous() reconstructs builder correctly
2. Round-trip preserves all information
3. Missing metadata handled gracefully
4. Invalid metadata raises clear errors

### High-Level Methods Tests

For each high-level method (with_dataset, with_learning_rate, etc.):
1. Applies correct override to config
2. Returns self for chaining
3. Validates input types
4. Works with all base config types

### Validation Tests

1. validate() calls tune validate correctly
2. Successful validation returns True
3. Failed validation raises with error message
4. Works with temp files
5. Works with saved configs

## Test Fixtures

### Sample Configs

```
tests/fixtures/
├── sample_data.json          # Minimal dataset for validation
├── torchtune_configs/        # Cached base configs for fast testing
│   ├── llama3_1_8B_lora.yaml
│   └── llama3_2_1B_lora.yaml
└── metadata_examples/        # Example metadata files
    ├── valid.meta.yaml
    └── invalid.meta.yaml
```

### Mock Run Logs

```
tests/fixtures/run_logs/
├── successful_runs.jsonl     # Logs from successful training runs
├── oom_failures.jsonl        # Logs from OOM errors
└── varied_configs.jsonl      # Diverse configs for learning tests
```

## Continuous Integration

**Test Matrix:**
- Python versions: 3.10, 3.11, 3.12
- OS: Linux (primary), macOS (optional)
- Torchtune versions: latest stable

**Test Commands:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=torchtune_config_writer --cov-report=html

# Run specific category
pytest tests/test_builder.py
pytest tests/test_integration.py -v

# Run fast tests only (skip integration)
pytest -m "not integration"
```

## Coverage Goals

**Focus on critical functionality, not arbitrary coverage metrics:**

**Must have 100% coverage:**
- Core merge logic (deep_merge function)
- Type compatibility checking
- Operation ordering
- Delete path navigation

**Should have high coverage (>90%):**
- Builder public API
- Config loading methods
- Metadata generation
- High-level methods

**Lower priority (<70% acceptable):**
- Example scripts
- CLI utilities (if added)
- Experimental features

## Test Development Workflow

**Pattern: Write tests alongside development**

1. **Red**: Write failing test for new feature
2. **Green**: Implement minimal code to pass
3. **Refactor**: Clean up, ensure all tests still pass
4. **Document**: Add docstring explaining what's tested

**Example:**
```python
# 1. RED - Write test first
def test_with_lora_rank():
    builder = TorchtuneConfigBuilder("llama3_1/8B_lora_single_device")
    builder.with_lora_rank(32)
    config = builder.build()
    assert config["model"]["lora_rank"] == 32

# 2. GREEN - Implement
def with_lora_rank(self, rank: int):
    return self.override({"model": {"lora_rank": rank}})

# 3. REFACTOR - Add validation
def with_lora_rank(self, rank: int):
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"lora_rank must be positive int, got {rank}")
    return self.override({"model": {"lora_rank": rank}})
```

## Special Considerations

### Integration Tests with Torchtune

**Issue**: Integration tests depend on `tune validate` which requires:
- Torchtune installation
- Valid model paths (may not exist in CI)

**Solution**:
1. Use `@pytest.mark.integration` decorator
2. Skip if torchtune not available: `@pytest.mark.skipif(not has_torchtune())`
3. Mock heavy operations in unit tests
4. Keep integration tests minimal and fast

### GPU Helper Tests

**Issue**: Learning mode requires historical run logs

**Solution**:
1. Provide fixture run logs in `tests/fixtures/`
2. Test both with/without historical data
3. Test bootstrap mode independently
4. Mock subprocess calls for log parsing

### Platform Differences

**Issue**: Path handling differs between Windows/Unix

**Solution**:
1. Use `pathlib.Path` everywhere
2. Test path resolution on multiple platforms
3. Normalize paths in assertions
4. Use `tmp_path` fixture from pytest for temp files

## Performance Testing

**Not required for MVP, but consider for future:**

1. Merge performance with deep nesting (1000+ levels)
2. Loading time for large configs (10MB+)
3. Memory usage for batch generation (1000+ configs)
4. Validation overhead for complex configs
