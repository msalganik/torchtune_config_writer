# LoRA Checkpoint Options: Full Model vs Adapter-Only

When fine-tuning with LoRA, you have two options for saving checkpoints. This guide explains the tradeoffs and how to configure each.

## Quick Comparison

| Aspect | Full Model Export | Adapter-Only Export |
|--------|-------------------|---------------------|
| **Storage per checkpoint** | ~15 GB | ~20-160 MB |
| **Setup complexity** | Simple | Medium |
| **Evaluation complexity** | Simple | Medium-Advanced |
| **Best for** | Small sweeps (< 5 models) | Large sweeps, production |
| **Inspect AI compatibility** | ✅ Direct support | ⚠️ Needs merge or custom loading |

## Option A: Full Model Export (Recommended for Getting Started)

### Configuration

```yaml
# experiment.yaml

controls:
  _override:
    checkpointer:
      _component_: torchtune.training.FullModelHFCheckpointer
      checkpoint_dir: ${output_dir}/checkpoints
      output_dir: ${output_dir}/hf_model
      # save_adapter_weights_only: False (default)

evaluation:
  model_format: "hf"  # HuggingFace full model
```

### What Gets Saved

```
results/
├── run_000/
│   └── hf_model/                    # ~15 GB
│       ├── model-00001-of-00002.safetensors
│       ├── model-00002-of-00002.safetensors
│       ├── config.json
│       ├── tokenizer.json
│       └── ...
└── run_001/
    └── hf_model/                    # ~15 GB
```

**Total for 4 runs**: ~60 GB

### Pros

- ✅ Simple setup - just works
- ✅ Direct Inspect AI support (`hf/path/to/model`)
- ✅ No merge/conversion needed
- ✅ Portable - can use anywhere HF models work

### Cons

- ❌ Large storage requirement
- ❌ Not practical for large sweeps (20+ models)
- ❌ Redundant - base model repeated in each checkpoint

### When to Use

- Small experiments (< 5 models)
- Quick prototyping
- When storage is not a concern
- When you want simplest workflow

---

## Option B: Adapter-Only Export (Recommended for Production)

### Configuration

```yaml
# experiment.yaml

controls:
  _override:
    checkpointer:
      _component_: torchtune.training.FullModelHFCheckpointer
      checkpoint_dir: ${output_dir}/checkpoints
      output_dir: ${output_dir}/adapter
      save_adapter_weights_only: True  # KEY: Only save adapter

evaluation:
  enabled: true
  model_format: "adapter"  # Changed from "hf"
  base_model: "meta-llama/Llama-3.1-8B"  # Base model for loading adapters

  # Choose evaluation strategy:
  merge_adapters: True   # Merge before eval (simple, more temp storage)
  # OR
  # merge_adapters: False  # Load directly (advanced, minimal storage)
```

### What Gets Saved

```
results/
├── run_000/
│   └── adapter/                     # ~20 MB (rank=8)
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       └── ...
├── run_001/
│   └── adapter/                     # ~40 MB (rank=16)
├── run_002/
│   └── adapter/                     # ~80 MB (rank=32)
└── run_003/
    └── adapter/                     # ~160 MB (rank=64)
```

**Total for 4 runs**: ~300 MB

### Storage Savings

- **4 models**: 60 GB → 300 MB (**200x reduction**)
- **16 models**: 240 GB → 1.2 GB (**200x reduction**)
- **64 models**: 960 GB → 4.8 GB (**200x reduction**)

### Evaluation Sub-Options

Since Inspect AI expects full models, you need to handle adapters. Two approaches:

#### Sub-Option B1: Merge Adapters (Simpler)

**How it works**:
1. Load base model
2. Load adapter
3. Merge → temporary full model
4. Evaluate full model
5. Delete temporary model

**Storage**:
- Persistent: ~20-160 MB (adapter)
- Temporary: ~15 GB (merged model during eval)

**Pros**:
- ✅ Simple - works with standard Inspect AI
- ✅ No custom code needed
- ✅ Reliable

**Cons**:
- ❌ Uses ~15 GB temp storage per eval job
- ❌ Slower (merge step adds time)

**Generated eval script** (example):
```python
# eval_000.py

from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base + adapter
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = PeftModel.from_pretrained(base, "results/run_000/adapter")

# Merge
merged = model.merge_and_unload()
merged.save_pretrained("/tmp/merged_run_000")

# Evaluate
eval(tasks, model="hf//tmp/merged_run_000", ...)

# Cleanup
shutil.rmtree("/tmp/merged_run_000")
```

#### Sub-Option B2: Load Adapters Directly (Advanced)

**How it works**:
1. Load base model
2. Load adapter with PEFT
3. Evaluate directly (no merge)

**Storage**:
- Persistent: ~20-160 MB (adapter)
- Temporary: ~15 GB (base model in memory, no disk)

**Pros**:
- ✅ Minimal storage
- ✅ Faster (no merge step)
- ✅ Can evaluate multiple adapters on same base model

**Cons**:
- ❌ Requires custom Inspect AI model provider
- ❌ Needs verification that Inspect AI supports PEFT
- ❌ More complex setup

**Generated eval script** (example):
```python
# eval_000.py

from peft import PeftModel
from inspect_ai.model import ModelAPI  # Custom provider needed

class PeftModelAPI(ModelAPI):
    def __init__(self, base_model, adapter_path):
        self.base = AutoModelForCausalLM.from_pretrained(base_model)
        self.model = PeftModel.from_pretrained(self.base, adapter_path)

    async def generate(self, input, config):
        # Use PEFT model directly
        return self.model.generate(...)

# Evaluate with PEFT model
eval(tasks, model=PeftModelAPI("meta-llama/Llama-3.1-8B", "results/run_000/adapter"), ...)
```

⚠️ **Note**: This approach requires verification that Inspect AI supports custom model providers or PEFT models.

### Pros (Overall)

- ✅ Massive storage savings (200x)
- ✅ Practical for large sweeps
- ✅ Only store what's unique (adapters)
- ✅ Faster training (smaller checkpoints)

### Cons (Overall)

- ❌ More complex evaluation setup
- ❌ Requires base model access during eval
- ❌ May need custom code for direct loading

### When to Use

- Large experiments (20+ models)
- Production workflows
- Limited storage budget
- SLURM clusters with storage quotas

---

## Implementation Guidance

### Phase 1: Start Simple

1. **Use Option A (Full Model)** for initial experiments
2. Validate your evaluation pipeline works
3. Measure actual storage usage

### Phase 2: Optimize

1. **Switch to Option B1 (Adapter + Merge)** if storage is an issue
2. Accept ~15 GB temp storage during eval
3. This is still a big win (60 GB → 300 MB persistent)

### Phase 3: Advanced (Optional)

1. **Try Option B2 (Direct Loading)** if you need to optimize further
2. Verify Inspect AI compatibility first
3. May require custom model provider code

---

## Tool Support

The `cruijff-kit generate` tool will support both options:

### Full Model (Default)

```yaml
evaluation:
  model_format: "hf"
```

Generates: Standard eval scripts with `hf/path/to/model`

### Adapter with Merge

```yaml
evaluation:
  model_format: "adapter"
  base_model: "meta-llama/Llama-3.1-8B"
  merge_adapters: True  # Default
```

Generates: Eval scripts that merge adapters before evaluation

### Adapter Direct (Advanced)

```yaml
evaluation:
  model_format: "adapter"
  base_model: "meta-llama/Llama-3.1-8B"
  merge_adapters: False
```

Generates: Eval scripts with custom PEFT model loading

---

## Recommendations

**For most users**: Start with **Option A (Full Model)**, then move to **Option B1 (Adapter + Merge)** when storage becomes an issue.

**For large-scale production**: Use **Option B1** from the start. Only attempt **Option B2** if you have:
- Storage/performance requirements that demand it
- Time to verify Inspect AI compatibility
- Willingness to maintain custom model loading code

---

## See Also

- `experiment.yaml` - Full model example
- `experiment_adapter_only.yaml` - Adapter-only example
- `eval_adapter_merge.py` - Example merge script
- `eval_adapter_direct.py` - Example direct loading script
- [Torchtune Checkpointing Docs](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [Inspect AI Models](https://inspect.aisi.org.uk/models.html)
