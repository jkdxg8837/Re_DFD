# Frame-count probe (spec §2.5) — LOCKED

Status: **LOCKED 2026-09-25 by the user: T = 21 (81 frames), per-GPU batch 1, grad accum 8, 2× A100-40GB.**
E4 / F2 must be re-probed at T=21 before use (caveat 1).

## Setup

- Hardware: 1 node, **2× NVIDIA A100-PCIE-40GB** (39.50 GiB usable each), driver 560.35.03. Slurm job 2217352 (gpu017).
- Script: `scripts/probe_frames.py` (driver + torchrun worker; the worker runs `train.py`'s `main` unchanged,
  except that `save_checkpoint` is a no-op). Sbatch: `runs/probe_frames/probe.sbatch`.
- Config = **E1a**: `fastgen/configs/experiments/WanT2V/config_dmd2.py` + README post-train overrides + decisions
  of 2026-09-24: `trainer.fsdp=True`, ViPE `long`, `batch_size_global=16`, `post_train=True`,
  `gen_real_replace_prob=0.5`, `gan_loss_weight_gen=0`, key map `{net: net}` (default),
  `pretrained_ckpt_path=dmd2_t2v_vipe/0001000`, `decoder_type=full`, seed 1.
- `trainer.max_iter=11` → iterations 1–10. Critic steps: 1–4 and 6–9. Student steps: 5 and 10.
  Memory is the same whether or not a student step replaces data. With GAN off, the teacher runs once per student
  step on `perturbed_data_teacher_score` in both cases, and the straight-through input keeps the same autograd
  graph (`runs/NOTES_code_reading.md` §4). So the probe does not need a step with replacement to measure the peak.
- Peak = `torch.cuda.max_memory_allocated()` over the whole run, per rank. Threshold: ≤ 90 % of total memory.

## Results

| latent T | pixel frames | per-GPU batch | grad accum | status | peak allocated (max over ranks) | peak reserved | % of 39.5 GiB | fits |
|---|---|---|---|---|---|---|---|---|
| 21 | 81 | 2 | 4 | **OOM** (both ranks, at iter 1–2: "tried to allocate 1.09 GiB, 715 MiB free") | 35.20 GiB | 38.36 GiB | 89.1 % (at OOM) | no |
| 21 | 81 | **1** | **8** | ok, 10 iters, 2 student + 7 logged critic steps, losses finite | **29.98 GiB** | 34.54 GiB | **75.9 %** | **yes** |

Other checks in the ok run: `net` loaded from `dmd2_t2v_vipe/0001000` (iteration 1000); `iter_start=0`;
`avg_vsd_loss` 0.398 (2 student steps), `avg_fake_score_loss` 0.134 (7 critic steps); host RAM ≈ 31.5 GB per rank.
Throughput ≈ **2.05 min per iteration** (8 micro-batches), i.e. ≈ 80 min for a 40-step run.

## Proposal

- **Lock T = 21 (81 frames, the DFD default) with per-GPU batch 1 × grad accum 8 = global 16.** This is §2.5
  saving step 1 only (semantics-preserving). No frame reduction is needed, so the length-consistency check and the
  extra T=21 E0 row are not needed.
- Gradient checkpointing is already on (`Wan(disable_grad_ckpt=False)`; §2.5 step 2 satisfied by default).
- Caveats:
  1. The probe covers E1a/E1b (E1b's warm-up is critic-only, so it needs less memory). Per §2.5, T is locked for the
     table from the most memory-hungry experiment. E4 inversion noise loading and F2 diagnostics (JVP /
     double gradient) do not exist yet. With 24 % headroom they will probably fit, but **they must be re-probed at
     T=21 before use**. If they do not fit, we have to re-lock and redo the table rows together.
  2. Per-GPU batch 1 changes the DFD replacement granularity (each all-or-nothing decision now covers 1 sample, not
     2; `runs/NOTES_code_reading.md` §2). The expected replacement rate (p = 0.5) is unchanged.
