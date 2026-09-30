#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=3850
#SBATCH -t 01:30:00
#SBATCH -J dfd_phase0
#SBATCH -o runs/phase0/phase0_gpu_%j.log
# Phase 0: (a) inspect released checkpoint keys, (b) render samples from dmd2 (E0) and ours (DFD official)
# with the README inference command on the first 4 web_dfd_eval_prompts, seeds 1 and 2.
set -x
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
source .venv/bin/activate
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python runs/phase0/inspect_ckpt_keys.py 2>&1 | tee runs/phase0/inspect_ckpt_keys.log
CK=checkpoints/wan_t2v_checkpoints_release
for name in ours_wan_t2v_vipe/0000040 dmd2_t2v_vipe/0001000; do
  tag=${name%%/*}
  for seed in 1 2; do
    torchrun --nproc_per_node=1 --standalone \
      scripts/inference/video_model_inference.py \
      --ckpt_path $CK/$name \
      --do_student_sampling True \
      --do_teacher_sampling False \
      --prompt_file runs/phase0/prompts_first4.txt \
      --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
      --save_as_gif False \
      --video_save_dir runs/phase0/samples/$tag/seed$seed - \
      trainer.seed=$seed trainer.fsdp=True trainer.ddp=False \
      log_config.name=phase0_inference_${tag}_seed$seed
    echo "EXIT $tag seed$seed: $?"
  done
done
echo PHASE0_GPU_DONE
