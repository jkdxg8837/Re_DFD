# Overrides for the E2 family (spec §3 E2, decisions 2026-09-28). Each arm = E1a + exactly the listed changes.
source runs/sbatch/common.sh
# COMMON minus its ViPE datatags, so each arm sets its data exactly once
COMMON_NODATA=(); DATA_VIPE=
for x in "${COMMON[@]}"; do
  if [[ $x == dataloader_train.datatags=* ]]; then DATA_VIPE=$x; else COMMON_NODATA+=("$x"); fi
done
DATA_E2='dataloader_train.datatags=["WDS:/gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation/data/mixkit_wds_v3"]'
DATA_E2S=(
  'dataloader_train.datatags=["WDS:/gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation/data/mixkit_vipe_pair_wds_v3"]'
  ++dataloader_train.key_map.student_real=student.mp4
  '++dataloader_train.video_keys=[real,student_real]'
  ++model.dfd_student_input_key=student_real
)
# F2 diagnostics (first three metrics), every student update; dir set per run
diag() { echo ++model.dfd_diagnostics_every=1 ++model.dfd_diagnostics_dir=/gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation/runs/$1/diag; }
# E3 (spec §3 E3): CFG uncond branch at the generated point on replaced batches
E3FLAG=++model.dfd_cfg_uncond_on_gen=True
# E4 (spec §3 E4): teacher-ODE inverted noise, 1,280-clip Mixkit subset with z_inv.npy
DATA_E4=(
  'dataloader_train.datatags=["WDS:/gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation/data/mixkit_inv_wds_v3"]'
  ++dataloader_train.key_map.z_inv=z_inv.npy
  ++model.dfd_inverted_noise_key=z_inv
)
