# Evaluation setup (spec §6, subset decision of 2026-09-24)

- Subset: `runs/eval/vbench_subset/` built by `scripts/eval/make_vbench_subset.py` (seed 0) from the official
  `VBench_full_info.json` (vbench 0.1.5, sha256 in `subset_meta.json`). 7 Quality-Score dimensions, 4 disjoint
  official prompt groups × 40 prompts = 160 prompts; 5 samples each → **200 videos per dimension, 800 per model**.
- Generation: `scripts/eval/gen_vbench_subset.sh <ckpt> <tag> [T]`, i.e. the README inference command with
  `--do_student_sampling True --do_teacher_sampling False`. Sample k uses `trainer.seed=k` (k = 0..4). The
  student sampler is deterministic (`ode`) and all prompts of one run start from the same seed-k noise, so every
  model gets identical initial noise (paired comparison). 832×480, 16 fps, `(T-1)*4+1` frames.
  Files: `runs/eval/<tag>/raw/seed<k>/...`, VBench names `runs/eval/<tag>/videos/<prompt>-<k>.mp4` (symlinks).
- Scoring: `.venv-vbench` (Python 3.10 via uv, torch 2.8.0+cu126, vbench 0.1.5, transformers 4.33.2),
  `scripts/eval/run_vbench.py` (standard mode, subset full_info), then `scripts/eval/vbench_quality_score.py`
  (official normalization + weights; dynamic degree weight 0.5). Model weights are cached in `data/vbench_cache`.
- Whole pipeline for one checkpoint: `sbatch --export=ALL,CKPT=...,TAG=...,T=... scripts/eval/eval_ckpt.sbatch`.
- Generation speed: ~13 s/video on A100-40GB (8.7 s sampling + VAE decode/save) → 800 videos ≈ 1 h 50 min on 2 GPUs.
- torch 2.8 defaults `torch.load(weights_only=True)`, which breaks VBench's AMT load; `run_vbench.py` sets
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (official VBench weights) and saves each dimension separately.
- Caveat: Quality Score on a 200-video-per-dimension subset is not directly comparable to leaderboard numbers
  (those use all official prompts). Compare models only within this subset.

| tag | ckpt | T | Slurm job |
|---|---|---|---|
| E0_dmd2_T21 | dmd2_t2v_vipe/0001000 | 21 | 2217323 (generation OK, VBench failed: weights_only) → 2217624 (VBench only) |
| REF_ours_T21 | ours_wan_t2v_vipe/0000040 | 21 | 2217324 (generation OK, VBench failed: weights_only) → 2217625 (VBench only) |
