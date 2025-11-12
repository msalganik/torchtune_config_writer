# Why Option B2 (Direct Adapter Loading) Needs Verification

Option B2 involves loading LoRA adapters directly in Inspect AI without merging. This sounds simple, but there are several technical uncertainties that need verification before using in production.

## What We Know for Certain

✅ **PEFT works with HuggingFace models**:
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = PeftModel.from_pretrained(base, "path/to/adapter")

# This works - PEFT wraps the base model with adapter
output = model.generate(...)
```

✅ **Inspect AI supports HuggingFace models**:
```python
from inspect_ai import eval

# This works - loads full HF model
eval(tasks, model="hf/meta-llama/Llama-3.1-8B", ...)
```

## What We DON'T Know (Need Verification)

### Uncertainty #1: Inspect AI's Model Loading Interface

**Question**: Can Inspect AI load a **pre-instantiated** PEFT model, or does it only accept **string identifiers**?

**What we see in docs**:
```python
# Docs show string-based model specification
eval(tasks, model="hf/model-name", ...)
```

**What we need to work**:
```python
# Can we pass a model object instead?
peft_model = PeftModel.from_pretrained(base, adapter_path)
eval(tasks, model=peft_model, ...)  # Does this work?
```

**Why it matters**:
- If Inspect AI only accepts strings, we can't pass pre-loaded PEFT models
- Would need to write a custom model provider (more complex)

---

### Uncertainty #2: Custom Model Provider Interface

**Question**: If we need a custom provider, what's the correct interface?

**What I showed in the example**:
```python
class PeftModelAPI(ModelAPI):
    async def generate(self, input: str, config: GenerateConfig):
        # Custom implementation
        ...
```

**Problems**:
1. ❓ Is `ModelAPI` the correct base class? (couldn't find current docs)
2. ❓ Is `generate()` the right method? (might be `__call__` or something else)
3. ❓ What's the signature of `GenerateConfig`? (what fields does it have?)
4. ❓ Do we need other methods? (tokenization, batching, error handling?)
5. ❓ How does Inspect AI instantiate custom providers? (registration? decorator?)

**Why it matters**:
- Wrong interface = code won't work
- Missing methods = runtime errors
- Interface may have changed between Inspect AI versions

---

### Uncertainty #3: HuggingFace Provider Implementation

**Question**: How does Inspect AI's `hf/` provider actually work under the hood?

**Likely implementation**:
```python
# Inside Inspect AI (hypothetical)
def load_hf_model(model_string):
    model_name = model_string.replace("hf/", "")
    return AutoModelForCausalLM.from_pretrained(model_name)
```

**What we don't know**:
1. ❓ Does it use `AutoModelForCausalLM`? (or something else?)
2. ❓ What parameters does it pass? (device_map, torch_dtype, etc.)
3. ❓ Does it cache models? (important for loading base model once)
4. ❓ Can we override or extend it? (to add PEFT loading)

**Specific concern**: Can we tell Inspect's HF provider to load an adapter?
```python
# Can we do something like this?
eval(tasks,
     model="hf/meta-llama/Llama-3.1-8B",
     adapter_path="results/run_000/adapter",  # Custom parameter?
     ...)
```

Probably not - would need custom provider.

---

### Uncertainty #4: PEFT Model Compatibility

**Question**: Do PEFT models behave identically to merged models in Inspect AI's inference pipeline?

**Potential issues**:

1. **Forward pass differences**:
   - PEFT models have slightly different forward pass (adapter routing)
   - Does this affect Inspect AI's generation logic?

2. **Generation parameters**:
   - Does PEFT respect all generation configs? (temperature, top_p, etc.)
   - Are there any parameter name differences?

3. **Batching**:
   - If Inspect AI batches inputs, does PEFT handle it correctly?
   - Are there performance implications?

4. **Edge cases**:
   - Streaming generation?
   - Constrained decoding?
   - Special tokens handling?

**Why it matters**:
- Even if we can load the model, inference might behave differently
- Results might not match merged model approach
- Could cause subtle bugs in evaluations

---

### Uncertainty #5: Model State and Context

**Question**: How does Inspect AI manage model state across multiple eval tasks?

**Scenario**:
```python
tasks = [
    mmlu_subset(),
    domain_qa(),
    hellaswag()
]

eval(tasks, model=peft_model, ...)
```

**What we don't know**:
1. ❓ Does Inspect AI reload the model between tasks?
2. ❓ Does it reset model state (clear caches, etc.)?
3. ❓ Are there thread safety concerns with PEFT models?
4. ❓ Memory management - does it unload/reload models?

**Why it matters**:
- PEFT models might have different state management requirements
- Could cause issues across multiple tasks
- Memory leaks or performance degradation

---

### Uncertainty #6: Practical Testing Gap

**Question**: Do the outputs actually match between direct loading and merging?

**What needs testing**:
```python
# Approach 1: Merge
merged = model.merge_and_unload()
results_merged = eval(tasks, model=merged, ...)

# Approach 2: Direct
results_direct = eval(tasks, model=peft_model, ...)

# Are these identical?
assert results_merged == results_direct  # ⚠️ Need to verify!
```

**Potential discrepancies**:
1. Numerical precision differences (unlikely but possible)
2. Generation parameters interpreted differently
3. Tokenization differences
4. Random seed handling

**Why it matters**:
- If results don't match, direct loading approach is invalid
- Even small differences could affect evaluation metrics
- Need to verify across multiple tasks and models

---

## What Verification Looks Like

To use Option B2 in production, you'd need to:

### Step 1: Check Inspect AI's Current Interface

```python
# Read current docs
# https://inspect.aisi.org.uk/reference/inspect_ai.model.html

# Check if custom model providers are supported
from inspect_ai.model import ModelAPI  # Does this exist?

# Check the interface
help(ModelAPI)  # What methods are required?
```

### Step 2: Write Minimal Test

```python
# test_peft_loading.py

from peft import PeftModel
from transformers import AutoModelForCausalLM
from inspect_ai import eval
from inspect_ai.dataset import Sample

# Load PEFT model
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
peft_model = PeftModel.from_pretrained(base, "results/run_000/adapter")

# Try to use it with Inspect AI
result = eval(
    [Sample(input="What is 2+2?", target="4")],
    model=peft_model,  # Can we pass model object?
)

print("✓ It works!" if result else "✗ Failed")
```

### Step 3: Compare Outputs

```python
# Load same adapter two ways
merged_model = peft_model.merge_and_unload()

# Run same eval with both
results_direct = eval(tasks, model=peft_model, ...)
results_merged = eval(tasks, model=merged_model, ...)

# Compare
for task_name in results_direct.keys():
    direct_score = results_direct[task_name]['score']
    merged_score = results_merged[task_name]['score']

    if abs(direct_score - merged_score) > 1e-6:
        print(f"⚠️ Mismatch in {task_name}: {direct_score} vs {merged_score}")
    else:
        print(f"✓ {task_name} matches")
```

### Step 4: Test Edge Cases

- Multiple tasks in sequence
- Large batches
- Different generation parameters
- Memory usage over time
- Multiple adapters on same base model

### Step 5: Document and Share

If it works:
```markdown
# Verified Configuration

✅ Tested with:
- Inspect AI version: 0.x.x
- PEFT version: 0.x.x
- Transformers version: 4.x.x

✅ Verified:
- Direct loading works with custom ModelAPI
- Results match merged model approach
- No memory leaks over 100 tasks
- Works with multiple adapters

Example code: [link to gist]
```

---

## Why Not Just Use Merge (Option B1)?

Valid question! Here's the tradeoff:

| Aspect | B1: Merge | B2: Direct |
|--------|-----------|------------|
| **Works today** | ✅ Yes | ❓ Unknown |
| **Temp storage** | 15 GB | 0 GB (just memory) |
| **Speed** | Slower (merge step) | Faster |
| **Code complexity** | Low | High |
| **Maintenance** | Low | Medium-High |
| **Risk** | Low | Medium |

**For most users**: B1 (merge) is the pragmatic choice:
- ✅ Guaranteed to work
- ✅ Only ~1-2 min merge time per eval
- ✅ Still saves 99% of persistent storage
- ✅ Simple code, easy to debug

**Only pursue B2 if**:
- You're running hundreds of evaluations
- Merge time adds up significantly
- You have time to do the verification work
- You can maintain custom model loading code

---

## Alternative: vLLM with LoRA

Worth mentioning: **vLLM has native LoRA adapter support**:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True
)

# Load adapter
llm.load_lora_adapter("run_000", "results/run_000/adapter")

# Generate
output = llm.generate("prompt", lora_request=LoRARequest("run_000", ...))
```

**If Inspect AI supports vLLM as a model provider**, this could be easier than custom HF provider.

Check Inspect AI docs for vLLM support:
```python
# Does this work?
eval(tasks, model="vllm/meta-llama/Llama-3.1-8B", ...)
```

If yes, you could extend it to support LoRA adapters.

---

## Bottom Line

**Option B2 needs verification because**:
1. Inspect AI's model loading interface is unclear
2. Custom model provider implementation is uncertain
3. PEFT compatibility with Inspect AI is unverified
4. No documented examples exist (as far as I found)
5. Results need to be validated against merged approach

**This doesn't mean it's impossible** - just that someone needs to:
- Read current Inspect AI docs carefully
- Test with real code
- Verify outputs match
- Document the working approach

Until then, **Option B1 (merge)** is the safe, proven path that gives you 99% of the benefits with 0% of the uncertainty.

---

## See Also

- [Inspect AI Models Documentation](https://inspect.aisi.org.uk/models.html)
- [Inspect AI Model API Reference](https://inspect.aisi.org.uk/reference/inspect_ai.model.html)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [vLLM LoRA Documentation](https://docs.vllm.ai/en/latest/features/lora.html)
