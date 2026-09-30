#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere_a100:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem-per-cpu=3850
#SBATCH -t 04:00:00
#SBATCH -J dfd_smoke
#SBATCH -o runs/smoke/smoke_%j.log
# Smoke tests (spec §0 rule 2: max_iter=20, save_ckpt_iter=10), locked T=21 / per-GPU batch 1 (runs/FRAMES_LOCK.md).
#   0. unit test tests/test_full_save_iter.py (CPU-only work, run inside this GPU job)
#   1. SMOKE_E1a         E1a config: net from dmd2_t2v_vipe/0001000, fake_score = teacher copy, GAN off;
#                        full_save_iter=20 -> iteration 10 is a net-only save, 20 (final) a full save
#   2. SMOKE_E1b_warmup  fake_score-only warm-up (student frozen: student_update_freq > max_iter)
#   3. SMOKE_E1b         post-train loading {net, fake_score} from the SMOKE_E1b_warmup checkpoint
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source .venv/bin/activate
python -m pytest tests/test_full_save_iter.py -v -p no:cacheprovider 2>&1 | tee runs/smoke/pytest_full_save_iter.log | tail -8
COMMON=(
  trainer.fsdp=True
  'dataloader_train.datatags=["WDS:/gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation/data/vipe_fliter_complete/long"]'
  trainer.batch_size_global=16
  dataloader_train.batch_size=1
  dataloader_train.shuffle_size=100
  dataloader_train.num_workers=2
  '+dataloader_train.decoder_type=full'
  model.post_train=True
  model.gen_real_replace_prob=0.5
  model.gan_loss_weight_gen=0
  log_config.wandb_mode=offline
  trainer.seed=1
  trainer.max_iter=20
  trainer.save_ckpt_iter=10
  trainer.logging_iter=10
)
CK=checkpoints/wan_t2v_checkpoints_release/dmd2_t2v_vipe/0001000
scripts/launch_run.sh SMOKE_E1a 2 "${COMMON[@]}" \
  trainer.checkpointer.pretrained_ckpt_path=$CK trainer.checkpointer.full_save_iter=20 log_config.name=SMOKE_E1a
scripts/launch_run.sh SMOKE_E1b_warmup 2 "${COMMON[@]}" \
  trainer.checkpointer.pretrained_ckpt_path=$CK model.student_update_freq=1000000 log_config.name=SMOKE_E1b_warmup
WU=FASTGEN_OUTPUT/fastgen/wan_dmd2/SMOKE_E1b_warmup/checkpoints/0000020
scripts/launch_run.sh SMOKE_E1b 2 "${COMMON[@]}" \
  trainer.checkpointer.pretrained_ckpt_path=$WU \
  '++trainer.checkpointer.pretrained_ckpt_key_map={net: net, fake_score: fake_score}' log_config.name=SMOKE_E1b
echo SMOKE_DONE
