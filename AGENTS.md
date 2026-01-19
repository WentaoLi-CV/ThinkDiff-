## Goal
Implement in ThinkDiff-main:
- DDP multi-GPU training via torchrun (my server-side verification)
- Dataset supports: caption + entities + modality + modality_id (+ sample_id), from medical WebDataset tar
- Entity-aware masking for medical text, preferably T5 span corruption (sentinel tokens / <extra_id_*>)
- Quantization first, before any LoRA
- LoRA finetuning on: vision encoder + text encoder + text decoder
- Per-rank (per-GPU) balanced batching for MoE: each rank local batch approx 1:1:1:1:1 modalities per step
- Replace vision FFN (prefer fc2) LoRA with ModalMoE, gated by modality_id via MoEContext



