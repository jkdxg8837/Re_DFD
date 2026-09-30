#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem-per-cpu=3850
#SBATCH -t 04:00:00
#SBATCH -J eval_resume
#SBATCH -o runs/eval/slurm_%x_%j.log
# Job 2222246 (E2s_seed1_it0040 eval): seed 3 generation died with SIGSEGV (0/160 videos), seeds 0,1,2,4 are fine.
# Regenerate seed 3 only, then run the rest of the eval pipeline (it skips generation once 800 videos exist).
set -euo pipefail
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
export CKPT=FASTGEN_OUTPUT/fastgen/wan_dmd2/E2s_seed1/checkpoints/0000040 TAG=E2s_seed1_it0040 T=21
mv runs/eval/$TAG/logs/seed3.log runs/eval/$TAG/logs/seed3_sigsegv_job2222246.log
SEEDS=3 scripts/eval/gen_vbench_subset.sh $CKPT $TAG $T
bash scripts/eval/eval_ckpt.sbatch
