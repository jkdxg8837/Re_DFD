#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem-per-cpu=3850
#SBATCH -t 01:30:00
#SBATCH -J dfd_extra
#SBATCH -o runs/eval/slurm_extra_%j.log
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
source .venv/bin/activate
python scripts/eval/extra_metrics.py E0_dmd2_T21 REF_ours_T21 && python scripts/eval/compare_extra.py E0_dmd2_T21 REF_ours_T21
