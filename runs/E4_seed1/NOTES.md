# E4_seed1: DFD + real data (Mixkit) + teacher-ODE inversion, spec §3 E4

Change vs E2_seed1: eps_student ← z_inv (teacher Wan2.1-1.3B, 50 Euler steps on the shift-5 grid, w=1, conditional on
the v3 caption, from the VAE posterior-mean latent at t=0 to t=0.999; `scripts/invert_real.py`);
`++model.dfd_inverted_noise_key=z_inv`, data `data/mixkit_inv_wds_v3` = fixed random 1,280-clip subset of the
5,227 Mixkit clips [CONFIRMED 2026-09-28]. E2 arm only.

Side effects (spec §1.1): the multi-step student input is forward_process(x, z_inv, t_student), so for t_student < 0.999
it is an interpolation between x and its own inverted noise — i.e. (approximately) a point on the teacher's ODE path
of x, not x noised with fresh Gaussian noise. The training latent x is a VAE posterior *sample* (trainer encode,
mode="sample") while z_inv was computed from the posterior *mean*; the mismatch is small but nonzero.
Data pool: 1,280 clips vs 5,227 for E2 (each run consumes 640 samples).

Hypothesis (before the run): inversion makes G(z_inv) land near x, shrinking delta_sq vs E2, so the teacher-score
replacement becomes a small, local correction (A.2 condition 1). If the loss of the diversity gain in E2 is caused
by large (x, G) gaps, E4 should recover part of it; if delta_sq shrinks but diversity does not recover, the gap is
not the cause. Risk: z_inv is not Gaussian (check z_std / reconstruction PSNR first), which moves the student input
off the noise distribution it sees at inference.

## Log
- 2026-09-29: inversion finished, 1,280/1,280 clips (jobs 2223372 / 2223373, 13 h each). Stats over all 1,280:
  z_std 0.847 (min 0.47, max 0.93), z_mean 0.002, corr(z_inv, x0) 0.19. Reconstruction check on 48 clips:
  latent MSE 0.0089, PSNR 32.2 dB (VAE-only ceiling 36.5 dB).
- 2026-09-30: the original `after_invert.sh` watcher had died before job 2223373 ended (empty log), so the WDS was
  not built for ~21 h. Resumed with `after_invert_resume.sh` (same steps, no squeue wait).
