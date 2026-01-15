## Goal
Implement on ThinkDiff-main:
- DDP multi-GPU training via torchrun
- Quantization 
- LoRA on vision encoder + text encoder + text decoder
- Replace vision FFN LoRA with ModalMoE (ThinkDiff-/thinkdiff/models/modalmoe)
- Dataset supports entities/modality/modality_id and distributed stratified batching with target ratios

## Rules
- Keep diffs minimal. No unrelated refactors.
- Must keep `bash runs/train_thinkdiff_clip.sh` runnable.
- Every milestone must run the debug config with torchrun and show rank0 logs.

## Verification (to be finalized)
- Debug DDP run: torchrun --nproc_per_node=2 train.py --cfg-path configs/debug_modalmoe_ddp.yaml
- Print: trainable params, modality counts, loss/aux_loss
