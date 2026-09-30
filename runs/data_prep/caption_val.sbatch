#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu-devel
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem-per-cpu=3850
#SBATCH -t 01:00:00
#SBATCH -J cap_val
#SBATCH -o runs/data_prep/slurm_%x_%j.log
# Prompt validation (spec §2.3): recaption 50 ViPE `long` videos with Qwen2.5-VL-7B and compare with the originals.
# Also a small prep_mixkit.py test (40 sources, separate output dir) on the CPUs while the GPU captions.
# Usage: sbatch --export=ALL,PV=v1 runs/data_prep/caption_val.sbatch
set -uo pipefail
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
PV=${PV:-v1}
export HF_HOME=$PWD/FASTGEN_OUTPUT/.cache
nvidia-smi --query-gpu=name,memory.total --format=csv
PY=.venv-vlm/bin/python
if [ "${PREP_TEST:-1}" = 1 ]; then
  $PY scripts/data_prep/prep_mixkit.py --src data/mixkit_src --out data/mixkit_prep_test --workers 12 --limit 40 \
    > runs/data_prep/prep_test.log 2>&1 &
  PREP=$!
fi
$PY scripts/data_prep/caption_qwen.py --list data/caption_val/vipe_list.txt \
  --out runs/data_prep/caption_val_${PV}.jsonl --prompt_version $PV --batch 8
$PY scripts/data_prep/compare_captions.py runs/data_prep/caption_val_${PV}.jsonl --out runs/data_prep/caption_val_${PV}.md
[ -n "${PREP:-}" ] && wait $PREP && tail -3 runs/data_prep/prep_test.log
echo VAL_DONE
