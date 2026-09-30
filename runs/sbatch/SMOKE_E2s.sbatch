#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere_a100:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem-per-cpu=3850
#SBATCH -t 01:30:00
#SBATCH -J sm_E2s
#SBATCH -o runs/SMOKE_E2s/slurm_%j.log
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
source runs/sbatch/e2_common.sh
scripts/launch_run.sh SMOKE_E2s 2 "${COMMON_NODATA[@]}" "${DATA_E2S[@]}" $(diag SMOKE_E2s) trainer.seed=1 \
  trainer.max_iter=20 trainer.save_ckpt_iter=10 trainer.logging_iter=5 \
  trainer.checkpointer.pretrained_ckpt_path=$CK log_config.name=SMOKE_E2s
