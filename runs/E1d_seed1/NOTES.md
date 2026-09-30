# E1d_seed1: E1a_seed1 + F2 diagnostics (synthetic reference for F2)

Identical to E1a_seed1 (ViPE `long`, seed 1, 40 steps) with `dfd_diagnostics_every=1`. Diagnostics are no-grad,
use their own RNG and do not change the loss or gradients (`tests/test_dfd_flags.py`), but FSDP/data-loader
nondeterminism means the trajectory is not bit-identical to E1a. Purpose: diagnostics only; no eval.
Only the final checkpoint is saved (`save_ckpt_iter=1000`).

## Result (2026-09-28)
Train job 2222247 (1 h 30 min), 128 diagnostic rows. x0 space: delta_sq 0.358, Var(Δ) 3.70e5, ‖E Δ‖² 3.8e2
(mean share 0.001), lipschitz_hat 2.01. vsd losses track E1a's. See runs/eval/diag_E1d_E2_E2s.md.
