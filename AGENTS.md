## Goal
Implement in ThinkDiff-main:
- DDP multi-GPU training via torchrun (my server-side verification)
- Dataset supports: caption + entities + modality + modality_id (+ sample_id), from medical WebDataset tar
- Entity-aware masking for medical text, preferably T5 span corruption (sentinel tokens / <extra_id_*>)
- Quantization first, before any LoRA
- LoRA finetuning on: vision encoder + text encoder + text decoder
- Per-rank (per-GPU) balanced batching for MoE: each rank local batch approx 1:1:1:1:1 modalities per step
- Replace vision FFN (prefer fc2) LoRA with ModalMoE, gated by modality_id via MoEContext

## Constraints / Rules
- Keep diffs minimal. No unrelated refactors.
- Must keep 'bash runs/debug_train_thinkdiff_clip_quantity.sh' runnable on my server.
- The Codex environment cannot access PyPI/GitHub reliably (pip/torchrun may fail). Treat Codex as OFFLINE code-editing:
  - DO NOT run pip installs or training commands in Codex.
  - Use ripgrep and file inspection to locate code, then implement changes.


