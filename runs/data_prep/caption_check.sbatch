#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu-devel
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=3850
#SBATCH -t 00:30:00
#SBATCH -J cap_check
#SBATCH -o runs/data_prep/slurm_%x_%j.log
# Caption vs Mixkit-title consistency check + F2 diag summary of SMOKE_E1d (both need torch/GPU, not the login node).
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
export HF_HOME=$PWD/FASTGEN_OUTPUT/.cache
.venv-vlm/bin/python scripts/data_prep/check_caption_title.py data/mixkit_prep/captions_v2.jsonl --out runs/data_prep/caption_title_check
.venv/bin/python scripts/eval/diag_summary.py runs/SMOKE_E1d/diag --out runs/SMOKE_E1d/diag_summary.md
echo CHECK_DONE
