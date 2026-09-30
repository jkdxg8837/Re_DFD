#!/bin/bash
# Generate the fixed VBench subset (runs/eval/vbench_subset/prompts.txt, 160 prompts) for one checkpoint:
# 5 samples per prompt = 5 inference runs with trainer.seed = 0..4 (sample index k uses seed k).
# The student sampler is deterministic ("ode"), and every prompt of one run starts from the same seed-k noise, so
# all models see identical initial noise (paired comparison). Seeds are spread over the visible GPUs.
#
# Usage (inside a Slurm allocation, from the repo root):
#   scripts/eval/gen_vbench_subset.sh <ckpt_prefix> <tag> [latent_T=21]
# Output: runs/eval/<tag>/raw/seed<k>/...mp4 and runs/eval/<tag>/videos/<prompt>-<k>.mp4 (symlinks, VBench naming)
set -euo pipefail
CKPT=$1
TAG=$2
T=${3:-21}
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
source .venv/bin/activate
OUT=runs/eval/$TAG
PROMPTS=runs/eval/vbench_subset/prompts.txt
mkdir -p $OUT/raw $OUT/logs
SEQ=$(( (T - 1) * 4 + 1 ))
echo "ckpt=$CKPT tag=$TAG T=$T ($SEQ frames)" | tee $OUT/gen_info.txt
NGPU=$(nvidia-smi -L | wc -l)
run_seed() {
  local k=$1 gpu=$2 rc=0
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nnodes=1 --nproc_per_node=1 --master_port=$((29500 + k)) \
    scripts/inference/video_model_inference.py \
    --ckpt_path $CKPT \
    --do_student_sampling True \
    --do_teacher_sampling False \
    --prompt_file $PROMPTS \
    --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
    --save_as_gif False \
    --video_save_dir $OUT/raw/seed$k - \
    trainer.seed=$k trainer.fsdp=True trainer.ddp=False \
    "model.input_shape=[16,$T,60,104]" \
    log_config.name=vbench_${TAG}_seed$k > $OUT/logs/seed$k.log 2>&1 || rc=$?
  echo "seed $k on gpu $gpu: exit $rc"
}
# round-robin the seeds over the GPUs; one process per GPU at a time. SEEDS (env, default "0 1 2 3 4") lets a
# failed seed be regenerated alone; the other seeds' videos are kept.
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2 3 4}"
for ((g = 0; g < NGPU; g++)); do
  ( for ((i = g; i < ${#SEED_LIST[@]}; i += NGPU)); do run_seed ${SEED_LIST[$i]} $g; done ) &
done
wait
python3 scripts/eval/link_vbench_names.py --prompts $PROMPTS --raw $OUT/raw --out $OUT/videos
