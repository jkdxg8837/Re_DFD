#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere_a100:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=3850
#SBATCH -t 08:00:00
#SBATCH -J dfd_eval
#SBATCH -o runs/eval/slurm_%x_%j.log
# Evaluate one checkpoint on the fixed VBench subset (spec §6): generate 160 prompts x 5 seeds, run the 7 quality
# dimensions, compute the Quality Score, then the saturation / seed-diversity extra metrics.
# Usage: sbatch --export=ALL,CKPT=<ckpt_prefix>,TAG=<tag>,T=<latent_T> scripts/eval/eval_ckpt.sbatch
set -euo pipefail
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
: "${CKPT:?}" "${TAG:?}"
T=${T:-21}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if [ "$(ls runs/eval/$TAG/videos 2>/dev/null | grep -c '\.mp4$')" -eq 800 ]; then
  echo "all 800 videos present for $TAG, skipping generation"
else
  scripts/eval/gen_vbench_subset.sh "$CKPT" "$TAG" "$T"
fi
export VBENCH_CACHE_DIR=$PWD/data/vbench_cache
CUDA_VISIBLE_DEVICES=0 .venv-vbench/bin/python scripts/eval/run_vbench.py \
  --videos runs/eval/$TAG/videos --name $TAG --out runs/eval/$TAG/vbench
python3 scripts/eval/vbench_quality_score.py runs/eval/$TAG/vbench/${TAG}_eval_results.json
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval/extra_metrics.py $TAG
echo EVAL_DONE
