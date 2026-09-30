# E1a_seed1 — DFD reproduction, fake_score = teacher copy (no warm-up)

## Setup
Start `dmd2_t2v_vipe/0001000` (net only, key map `{net: net}`); fake_score = HF teacher copy; GAN off
(`gan_loss_weight_gen=0`); ViPE `long`; `post_train=True`, `p=0.5`; T=21, 2× A100 × batch 1 × accum 8 = 16;
`max_iter=41` (iterations 1–40 → 8 generator + 32 critic updates, like `ours_wan_t2v_vipe/0000040`);
saves every 5 (net-only), full at 20/40 (+ duplicate final 41); `trainer.seed=1`.

## Hypothesis (before the run)
- If the released DFD result mainly comes from the data-forcing term acting on the student, E1a should move the
  saturation metrics from E0 (frac sat>0.8 = 32.8 %) toward REF (3.0 %) within 40 iterations. VBench quality should
  stay within noise of both.
- Risk (spec §5.1): a teacher-copy fake_score has never tracked the DMD2 student, so early VSD gradients are
  `fake(x_t^gen) − teacher_cfg(x_t^real)` with fake ≈ teacher without CFG. That is a large, CFG-dominated
  difference, which may push the student somewhere different from the official run. Expect E1a ≠ REF if critic
  state matters.
- Student input distribution: ViPE noised samples at t ∈ {0.999, 0.937, 0.833, 0.624}, same as the DMD2 run that
  produced the start (no change of input distribution in E1).

## Training (Slurm 2220976, 2026-09-26, 1 h 26 min, exit 0)
- Load report: net (dmd2 0001000) / fake_score (HF teacher copy) / teacher (HF): missing 0, unexpected 0; no discriminator.
- `0000040.pth` scheduler state: net `last_epoch` 8, fake_score 32, no discriminator — **identical to the official
  `ours_wan_t2v_vipe/0000040.pth`**, so the update schedule matches the authors' run.
- vsd_loss at the 8 generator steps (iters 5…40): 0.423, 0.478, 0.502, 0.513, 0.463, 0.756, 0.813, 0.408 (single
  micro-batch-averaged values, noisy). fake_score_loss per logging window: 0.02–0.16. No NaN/Inf.
- Checkpoints: net-only at 5/10/15/25/30/35 (2.7 GB each); full at 20, 40 and 41 (41 = same weights as 40, extra final save).
- Training-time samples (`runs/E1a_seed1/frames/strip_generation.png`) use a different caption at every step, so they
  are only a sanity check (videos render fine), not a comparison.

## Conclusion (after the run, 2026-09-27)
See `RESULTS.md` (E1a_seed1 trajectory). Saturation drops monotonically (frac sat>0.8 32.8 → 11.6 %) and plateaus after
~30 iterations, still far above REF (3.0 %). Diversity rises significantly above E0 (+0.014) and ends ≥ REF (n.s.).
VBench Quality flat within noise. **Reproduction not achieved on saturation** → hypothesis "the data-forcing term alone,
with a teacher-copy critic, reproduces the released DFD result in 40 iterations" is not supported. The critic state
(spec §5 risk 1) is the leading suspect; training-seed noise is untested.
