#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere_a100:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem-per-cpu=3850
#SBATCH -t 01:30:00
#SBATCH -J probe_E4
#SBATCH -o runs/probe_frames_E4/probe_%j.log
# Memory re-probe at the locked T=21 / per-GPU batch 1 (runs/FRAMES_LOCK.md caveat: re-probe before F2) with the
# E4 config (inverted-noise stream z_inv) + F2 diagnostics.
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
source .venv/bin/activate
source runs/sbatch/e2_common.sh
export WANDB_MODE=offline
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python scripts/probe_frames.py --ngpu 2 --ladder 21 --batch_sizes 1 --out runs/probe_frames_E4 -- \
  "${COMMON_NODATA[@]}" "${DATA_E4[@]}" $(diag probe_frames_E4) \
  trainer.checkpointer.pretrained_ckpt_path=$CK trainer.seed=1
echo PROBE_DONE
