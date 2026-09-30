#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere_a100:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem-per-cpu=3850
#SBATCH -t 16:00:00
#SBATCH -J invert
#SBATCH -o runs/E4_seed1/slurm_invert_%j.log
# Teacher-ODE inversion for E4 (50 Euler steps, w=1) of LIST, one process per GPU; resumable.
# Usage: sbatch --export=ALL,LIST=...,OUT=...,RECON=25[,SHARD0=0,NSHARDS=4] runs/E4_seed1/invert.sbatch
# (global shard = SHARD0 + local GPU index; default one job covering all shards on its GPUs)
set -uo pipefail
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
source .venv/bin/activate
: "${LIST:?}" "${OUT:?}"
NGPU=$(nvidia-smi -L | wc -l)
SHARD0=${SHARD0:-0}; NSHARDS=${NSHARDS:-$NGPU}
for ((g = 0; g < NGPU; g++)); do
  CUDA_VISIBLE_DEVICES=$g torchrun --nnodes=1 --nproc_per_node=1 --master_port=$((29600 + SHARD0 + g)) scripts/invert_real.py \
    --list $LIST --out $OUT --shard $((SHARD0 + g)) --nshards $NSHARDS --recon ${RECON:-0} \
    --config fastgen/configs/experiments/WanT2V/config_dmd2.py - trainer.fsdp=True trainer.ddp=False \
    log_config.name=invert_mixkit_s$((SHARD0 + g)) > $OUT/invert_s$((SHARD0 + g)).log 2>&1 &
done
wait
cat $OUT/meta_s*.jsonl | wc -l
echo INVERT_DONE
