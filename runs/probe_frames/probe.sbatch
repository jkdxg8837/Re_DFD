#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere_a100:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem-per-cpu=3850
#SBATCH -t 06:00:00
#SBATCH -J dfd_probe
#SBATCH -o runs/probe_frames/probe_%j.log
# Frame-count probe (spec §2.5) with the E1a config: README post-train command + decisions of 2026-09-24
# (gan_loss_weight_gen=0, key_map {net: net} = FastGen default, 2x A100-40GB, global batch 16).
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
source .venv/bin/activate
until grep -q '^DONE' runs/phase0/download_vipe_long.log; do echo "waiting for ViPE download..."; sleep 60; done
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export WANDB_MODE=offline
python scripts/probe_frames.py --ngpu 2 --out runs/probe_frames -- \
  trainer.fsdp=True \
  'dataloader_train.datatags=["WDS:/gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation/data/vipe_fliter_complete/long"]' \
  trainer.batch_size_global=16 \
  dataloader_train.shuffle_size=100 \
  dataloader_train.num_workers=2 \
  trainer.checkpointer.pretrained_ckpt_path=checkpoints/wan_t2v_checkpoints_release/dmd2_t2v_vipe/0001000 \
  '+dataloader_train.decoder_type=full' \
  model.post_train=True \
  model.gen_real_replace_prob=0.5 \
  model.gan_loss_weight_gen=0 \
  log_config.wandb_mode=offline \
  trainer.seed=1
echo PROBE_DONE
