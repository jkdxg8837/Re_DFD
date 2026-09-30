#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu-devel
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem-per-cpu=3850
#SBATCH -t 00:20:00
#SBATCH -J diag_sum
#SBATCH -o runs/eval/slurm_%x_%j.log
# F2 diagnostics summary for E1d / E2 / E2s (torch needed -> GPU job, not the login node)
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
.venv/bin/python scripts/eval/diag_summary.py ${DIRS:-runs/E1d_seed1/diag runs/E2_seed1/diag runs/E2s_seed1/diag} --out ${OUTMD:-runs/eval/diag_E1d_E2_E2s.md}
echo DIAG_DONE
