# Archive: Phase 2+ Examples

This directory contains examples and documentation for features planned for Phase 2 and beyond.

## Contents

### lora_rank_with_eval/
Complete example showing torchtune config generation + Inspect AI evaluation integration.

**Status**: Phase 2+ feature
- Demonstrates evaluation config generation
- Shows adapter options for evaluation
- Includes complete end-to-end workflow

**Why archived**: Phase 1 focuses solely on torchtune config generation. Evaluation integration is deferred to Phase 2 to ensure we build on a solid foundation.

**When to use**: After Phase 1 is complete and stable, these examples will inform Phase 2 implementation of evaluation integration.

---

## Phase 1 vs Phase 2

**Phase 1 (Current):**
- experiment.yaml → torchtune configs
- CLI and Python API
- Variables and controls
- Deep merge semantics
- Pydantic validation

**Phase 2 (Planned):**
- Evaluation config generation (Inspect AI)
- SLURM script generation
- Metadata tracking
- Enhanced Python API
- Examples from this directory

---

See [SPEC.md](../../SPEC.md) for current Phase 1 scope.
