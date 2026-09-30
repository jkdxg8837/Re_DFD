# Smoke tests (Slurm 2220067, 2026-09-26) — PASSED

Config: E1a/E1b settings, locked T=21, per-GPU batch 1 × accum 8 × 2 A100-40GB, `max_iter=20` (iterations 1–19 run),
`save_ckpt_iter=10`, `logging_iter=10`. Per-run folders: `runs/SMOKE_E1a`, `runs/SMOKE_E1b_warmup`, `runs/SMOKE_E1b`
(cmd.sh, config_dump.yaml, git_diff.patch, train.log, load_report.json).

| check | SMOKE_E1a | SMOKE_E1b_warmup | SMOKE_E1b |
|---|---|---|---|
| unit test `tests/test_full_save_iter.py` | 3 passed | | |
| net source / missing / unexpected | dmd2_t2v_vipe/0001000 / 0 / 0 | dmd2_t2v_vipe/0001000 / 0 / 0 | warmup ckpt 0000020 / 0 / 0 |
| fake_score source / missing / unexpected | HF teacher copy / 0 / 0 | HF teacher copy / 0 / 0 | warmup ckpt 0000020 / 0 / 0 |
| teacher (HF) missing / unexpected | 0 / 0 | 0 / 0 | 0 / 0 |
| discriminator | not built (GAN off) | not built | not built |
| params per module | 1.4190 B | 1.4190 B | 1.4190 B |
| losses at iter 10 (avg) | vsd 0.464 (2 student steps), fake_score 0.130 | fake_score 0.089, no vsd (student frozen ✓) | vsd 0.343, fake_score 0.167 |
| NaN / Inf in log | none | none | none |
| checkpoints | 0000010 **net-only** (2.7 GB) + 0000020 full (16 GB) ✓ `full_save_iter` | 0000010 + 0000020 full | 0000010 + 0000020 full |
| sample videos (wandb offline media) | generation/real at iter 1 and 10, 81×832×480 @16 fps, render fine | same | same |
| loop time | 38.3 min / 19 it (2.0 min/it) | 36.7 min / 19 it (1.9 min/it) | 38.4 min / 19 it |

Notes
- The load report is taken before FSDP wrapping, so it shows fp32 params. `fastgen/utils/distributed/fsdp.py` then
  casts to `model.precision` = bf16 (`precision_amp=None`), and saved `net_model` is bf16 (2.7 GB, same as the release).
- Checkpoint sizes: full = net 2.7 + fake_score 2.7 + net_optim 5.3 + fake_score_optim 5.3 ≈ 16 GB; net-only 2.7 GB.
- Loop semantics: `range(1, max_iter)` runs iterations 1..max_iter−1, then saves once more at `max_iter`. The official
  `0000040` has 8 net updates (.pth: net scheduler step 8), i.e. iteration 40 ran and the ckpt was saved in-loop
  → main runs use `max_iter=41` (the extra final save `0000041` duplicates 0000040's weights).
- Sample frames: `runs/smoke/frames/`.
- Disk used by smoke checkpoints: ≈ 83 GB (deleted 2026-09-26 with user approval, see below).
- 2026-09-26: smoke checkpoints (83 GB) deleted with the user's approval to stay under the GPFS soft quota; logs, configs, load reports and wandb sample media kept.
