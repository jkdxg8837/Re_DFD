# E1b_warmup_seed1 — fake_score-only warm-up for E1b

## Setup
Same as E1a but `model.student_update_freq=1000000` (student frozen: no generator step in iterations 1..200),
`max_iter=201` → 200 critic updates, one full save at 0000201 (`save_ckpt_iter` disabled); `trainer.seed=1`.
N=200 (decision 2026-09-24).

## Hypothesis
fake_score_loss decreases from the teacher-copy level and plateaus, i.e. the fake score adapts to the DMD2 student's
distribution. The official critic had 800 DMD2-phase updates, so 200 may not fully match it.

## Conclusion
_pending_

## Status
DEFERRED (2026-09-26): the user decided not to warm up fake_score for now; Slurm jobs 2220977 / 2220978 and the 8 E1b evals (2220987–2220994) were cancelled before starting. Keep for later work (`runs/sbatch/E1b_warmup_seed1.sbatch`).
