# E1b_seed1 — DFD reproduction with warmed-up fake_score

## Setup
As E1a, but `pretrained_ckpt_path = E1b_warmup_seed1/checkpoints/0000201` with key map
`{net: net, fake_score: fake_score}` (net unchanged = dmd2_t2v_vipe/0001000 weights; fake_score after 200 critic
updates). Optimizers fresh. `trainer.seed=1`.

## Hypothesis
With a critic that tracks the student, the post-train dynamics are closer to the authors' run (their critic had
tracked the student). E1b should be closer to REF than E1a on saturation and VBench.

## Conclusion
_pending_

## Status
DEFERRED (2026-09-26): the user decided not to warm up fake_score for now; Slurm jobs 2220977 / 2220978 and the 8 E1b evals (2220987–2220994) were cancelled before starting. Keep for later work (`runs/sbatch/E1b_seed1.sbatch`).
