#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere_a100:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem-per-cpu=3850
#SBATCH -t 04:00:00
#SBATCH -J E1b_s1
#SBATCH -o runs/E1b_seed1/slurm_%j.log
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
source runs/sbatch/common.sh
WU=FASTGEN_OUTPUT/fastgen/wan_dmd2/E1b_warmup_seed1/checkpoints/0000201
scripts/launch_run.sh E1b_seed1 2 "${COMMON[@]}" trainer.seed=1 \
  trainer.max_iter=41 trainer.save_ckpt_iter=5 trainer.checkpointer.full_save_iter=20 trainer.logging_iter=5 \
  trainer.checkpointer.pretrained_ckpt_path=$WU \
  '++trainer.checkpointer.pretrained_ckpt_key_map={net: net, fake_score: fake_score}' log_config.name=E1b_seed1
