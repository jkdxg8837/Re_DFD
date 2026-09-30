# DFD real-data experiments (based on csy2077/data-forcing-distillation / FastGen)

## What this is
Reproduce DFD post-training starting from DFD's released Wan2.1-1.3B DMD2 checkpoint (`dmd2_t2v_vipe/0001000`), then study why swapping synthetic data for real-world video fails and how to fix it. No FastWan.
Full spec: `DFD_EXPERIMENTS.md`. **Before any task, read the sections relevant to it**; if unsure which, read §0 and §8.

## Language
Reply to me in Chinese. Write files, code comments and commit messages in English.

## Current status (Claude updates this after each step)
- Stage: DFD re-implementation DONE (E1a_seed1 = baseline, 2026-09-27). E2 family DONE (2026-09-28): Mixkit data (5,227 clips, Qwen2.5-VL-7B prompt v3; v2 rejected), E2 / E2s / E1d trained 40 steps, light evals it20/it40, F2 diagnostics (first 3 metrics). E3/E3s DONE (2026-09-29, RESULTS.md). E4: inversion 1,280/1,280 done (2026-09-29, jobs 2223372/73; recon 32.2 dB vs VAE 36.5 dB on 48 clips, z_std 0.85, corr(z_inv, x0) 0.19); 2026-09-30: building data/mixkit_inv_wds_v3 (after_invert_resume.sh; original watcher died) -> memory probe -> SMOKE_E4 -> E4_seed1 + light evals
- Locked frame count: T=21 (81 frames), per-GPU batch 1 x accum 8 x 2 A100-40GB, locked 2026-09-25 (runs/FRAMES_LOCK.md); re-probe before E4/F2
- Latest conclusion: E2 (Mixkit) loses DFD's diversity gain (≈ E0) but keeps de-saturation and gains motion; E2s does not restore diversity. E3 (CFG uncond at gen) does not help (div ≈ E0, −Quality, −motion vs E2); E3s is harmful (div 0.271 < E0, Dyn. Deg. 0.725). F2: E2 ≈ E1d; E2s/E3s ~4× delta_sq. Single seed. See RESULTS.md

## Hard rules
- Never change settings tagged `[VERIFIED]` or `[CONFIRMED]` in `DFD_EXPERIMENTS.md`; for `[ASK ME]`, stop and ask before first use.
- New features go behind config flags whose defaults reproduce the original repo; never modify existing logic branches.
- Smoke test first (`trainer.max_iter=20`). Run real training in the background with `nohup` or `tmux`, log to `runs/<EXP_ID>/train.log`, check progress with `tail`; never block in the foreground.
- Probe frame count only before training (§2.5). OOM mid-training: stop and report; never resume with a different frame count.
- After loading a checkpoint, print missing / unexpected keys per submodule; report any non-empty result.
- Never delete data or checkpoints; if disk runs low, stop and ask. Exception (user, 2026-09-28): once a smoke test has passed, delete its `FASTGEN_OUTPUT/.../SMOKE_*/checkpoints/` (keep logs, config, load_report, wandb) and note it in the run notes.
- Never submit CPU-only Slurm jobs (e.g. `-p devel` / `-p compute`); they are not allowed on this server. Every Slurm job must request a GPU (`-p gpu` or `gpu-devel`, account `su007-hw-gpu`, `--gres=gpu:ampere_a100:N`). CPU-only checks (unit tests, key inspection) go inside a GPU job, or on the login node only if they don't import torch (~540 MB memory cap there).

- Heavy evaluation (multiple training seeds, full eval of every checkpoint) is only for the E0/E1 baseline. Later experiments (E2–E5, F-series): single training seed and a light eval; don't queue full sweeps for them.

## Where state lives
- Phase 0 findings: `runs/PHASE0.md`
- Code-reading notes: `runs/NOTES_code_reading.md`
- Frame probe: `runs/FRAMES_LOCK.md`
- Per experiment: `runs/<EXP_ID>/` (cmd.sh, config_dump.yaml, git_diff.patch, train.log, NOTES.md)
- Results table: `RESULTS.md`
- Moving to another server: `MIGRATION.md` (layout, envs, what to copy, status)

At the start of a new session, read these files to recover progress before continuing.