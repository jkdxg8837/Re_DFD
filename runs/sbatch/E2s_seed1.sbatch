#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere_a100:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem-per-cpu=3850
#SBATCH -t 04:00:00
#SBATCH -J E2s_s1
#SBATCH -o runs/E2s_seed1/slurm_%j.log
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
source runs/sbatch/e2_common.sh
scripts/launch_run.sh E2s_seed1 2 "${COMMON_NODATA[@]}" "${DATA_E2S[@]}" $(diag E2s_seed1) trainer.seed=1 \
  trainer.max_iter=41 trainer.save_ckpt_iter=5 trainer.checkpointer.full_save_iter=20 trainer.logging_iter=5 \
  trainer.checkpointer.pretrained_ckpt_path=$CK log_config.name=E2s_seed1
