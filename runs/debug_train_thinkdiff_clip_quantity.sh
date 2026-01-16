gpu_id=$1
export CUDA_VISIBLE_DEVICES=$gpu_id
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

torchrun --nproc-per-node $gpu_num -m train --cfg-path configs/debug_modalmoe_ddp.yaml
