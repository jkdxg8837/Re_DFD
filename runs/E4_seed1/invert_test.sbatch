#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu-devel
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=3850
#SBATCH -t 01:00:00
#SBATCH -J inv_test
#SBATCH -o runs/E4_seed1/slurm_invert_test_%j.log
# 3-clip inversion + reconstruction test (timing and quality) before the full run.
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
source .venv/bin/activate
mkdir -p data/mixkit_inv_test
time torchrun --nnodes=1 --nproc_per_node=1 --master_port=29650 scripts/invert_real.py \
  --list data/mixkit_inv/test3.txt --out data/mixkit_inv_test --recon 3 \
  --config fastgen/configs/experiments/WanT2V/config_dmd2.py - trainer.fsdp=True trainer.ddp=False \
  log_config.name=invert_test > data/mixkit_inv_test/invert.log 2>&1
tail -5 data/mixkit_inv_test/invert.log; cat data/mixkit_inv_test/meta_s0.jsonl
echo TEST_DONE
