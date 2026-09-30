# Shared overrides for E1 runs (README post-train command + decisions 2026-09-24/25; runs/FRAMES_LOCK.md)
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
)
CK=checkpoints/wan_t2v_checkpoints_release/dmd2_t2v_vipe/0001000
