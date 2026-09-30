# Results

Format: spec §7 (`Dfd experiments.md`). Eval protocol: spec §6 (VBench, 5 samples/prompt, fixed seeds, 480×832, 16 fps).
All rows in one table use the same locked latent T (`runs/FRAMES_LOCK.md`).

| Model Type | Data | T (latent) | Steps | VBench Quality | Motion Smoothness | Dynamic Degree | ckpt | Notes |
|---|---|---|---|---|---|---|---|---|
| Original Model (E0) | – | 21 | 0 | 0.8479 | 0.9828 | 0.820 | dmd2_t2v_vipe/0001000 | VBench subset (200 videos/dim), `runs/eval/E0_dmd2_T21` |
| DFD official (ref) | ViPE long | 21 | 40 | 0.8499 | 0.9838 | 0.860 | ours_wan_t2v_vipe/0000040 | GAN off per .pth (runs/PHASE0.md §3); `runs/eval/REF_ours_T21` |
| + DFD (E1a) | ViPE long | 21 | 40 | 0.8500 | 0.9835 | 0.805 | E1a_seed1/0000040 | reproduction, fake_score = teacher copy; sat>0.8 = 11.6 % vs REF 3.0 %; **accepted as the DFD baseline by the user (2026-09-27)**, gap attributed to different data |
| + DFD (E1b) | ViPE long | | 40 | | | | | deferred (fake_score warm-up variant, user 2026-09-26) |
| + DFD, + real-world data (E2) | Mixkit (5,227 clips, Qwen2.5-VL-7B v3 captions) | 21 | 40 | 0.8534 | 0.9835 | 0.920 | E2_seed1/0000040 | single seed, light eval (it20/it40). vs E1a it40: div DINOv2 **−0.0186\*** (E1a's +0.014 gain over E0 lost; vs E0 −0.004 n.s.), sat>0.8 −2.5 pp\* (9.1 %), Dyn. Deg. +0.115\*, Quality +0.003 n.s. |
| + DFD, real data on the teacher side only (E2s) | student input ViPE `long` (caption-NN paired), replacement Mixkit | 21 | 40 | 0.8469 | 0.9848 | 0.905 | E2s_seed1/0000040 | vs E1a it40: div −0.0131\* (≈ E0, +0.001), sat>0.8 −0.6 pp\* (11.0 %); vs E2 it40: div +0.005 n.s., Quality −0.0066\*, sat +1.9 pp\* |
| + DFD, + real-world data, + CFG (E3) | Mixkit (E2 arm) | 21 | 40 | 0.8452 | 0.9838 | 0.825 | E3_seed1/0000040 | uncond branch at gen point on replaced batches. vs E2 it40: div +0.004 n.s. (≈ E0), Quality −0.0082\*, Dyn. Deg. −0.095\* (Mixkit's motion gain lost), sat>0.8 +0.2 n.s. (9.3 %); at it20 de-saturation slower (22.9 % vs 14.2 %) |
| + DFD, real data teacher-side only, + CFG (E3s) | ViPE student input, Mixkit replacement | 21 | 40 | 0.8427 | 0.9887 | 0.725 | E3s_seed1/0000040 | vs E2s it40: div **−0.0248\*** (0.2714, below E0 by −0.024\*), Dyn. Deg. −0.180\*, sat>0.8 +8.5 pp\* (19.5 %), Motion Sm. +0.004\*: mode-seeking / less motion |
| + DFD, + real-world data, + inverse (E4) | <dataset> | | | | | | | |
| + DFD, + real-world data, + new loss (E5) | <dataset> | | | | | | | formula pending |

## F2 diagnostics

| Experiment | reg_mean_norm | reg_var | delta_sq | lipschitz_hat | B_hat | cos(g_DFD, g_DMD) |
|---|---|---|---|---|---|---|
| E1d (= E1a + diag, ViPE) | 3.8e2 (mean share 0.001) | 3.70e5 | 0.358 | 2.01 | not implemented | not implemented |
| E2 (Mixkit) | 2.5e2 (0.001) | 3.36e5 | 0.309 | 2.13 | – | – |
| E2s (ViPE student input, Mixkit replacement) | 1.5e3 (0.002) | 8.97e5 | 1.269 | 1.72 | – | – |
| E3 (E2 + CFG variant; standard-CFG Δ) | 2.9e2 (0.001) | 3.31e5 | 0.357 | 2.10 | – | – |
| E3s (E2s + CFG variant; standard-CFG Δ) | 9.3e2 (0.001) | 9.27e5 | 1.209 | 1.83 | – | – |

x0 space (Δ·σ²/α = teacher-x0 difference minus (x − G)), pooled over all 8 student updates × 16 samples, CFG-guided
teacher; per-step and per-t-bin tables in `runs/eval/diag_E1d_E2_E2s.md`. E1d and E2 share seeds, hence identical
t draws (paired). No monotone trend over the 8 student updates in any run (16 samples per update, noisy).
Only the first three metrics are implemented (spec §4, 2026-09-28).

## Notes

### 2026-09-25 — E0 vs official DFD on the VBench subset (T=21)

Paired bootstrap over prompts (10k replicates, same prompts and seeds for both models;
`scripts/eval/paired_bootstrap.py` → `runs/eval/bootstrap_E0_vs_REF_T21.txt`). REF − E0:

| metric | E0 | REF | REF−E0 | 95 % CI |
|---|---|---|---|---|
| Quality Score | 0.8479 | 0.8499 | +0.0020 | [−0.0033, +0.0073] |
| subject consistency | 0.9643 | 0.9631 | −0.0012 | [−0.0044, +0.0026] |
| background consistency | 0.9481 | 0.9372 | **−0.0109** | [−0.0148, −0.0072] |
| temporal flickering | 0.9799 | 0.9788 | −0.0010 | [−0.0033, +0.0014] |
| motion smoothness | 0.9828 | 0.9838 | +0.0009 | [−0.0002, +0.0022] |
| dynamic degree | 0.820 | 0.860 | +0.040 | [−0.020, +0.100] |
| aesthetic quality | 0.6528 | 0.6458 | −0.0070 | [−0.0148, +0.0013] |
| imaging quality | 0.6651 | 0.6808 | **+0.0157** | [+0.0049, +0.0267] |

- On this subset, the Quality Score **does not separate** the DMD2 start from the official DFD result (the
  difference is inside the prompt-sampling noise, CI half-width ≈ 0.005). Only imaging quality (+) and background
  consistency (−) move significantly.
- The large saturation difference seen in Phase 0 (26 % vs 1 % of pixels with saturation > 0.8) is barely reflected
  by these 7 dimensions, and diversity is not measured by any of them.
- Consequence for the E1 reproduction check (spec §3): "every VBench gap within seed noise" is a weak test here,
  because E0 itself is within noise of REF. A discriminative check needs a saturation / diversity metric in
  addition (proposal sent to the user).

### 2026-09-25 — extra metrics, E0 vs official DFD (T=21, same 800 subset videos)

`scripts/eval/extra_metrics.py` + `scripts/eval/compare_extra.py` (paired bootstrap over 160 prompts) →
`runs/eval/extra_E0_dmd2_T21_vs_REF_ours_T21.json`.

| metric | E0 | REF | REF−E0 | 95 % CI |
|---|---|---|---|---|
| saturation mean | 0.5719 | 0.3604 | **−0.2115** | [−0.2231, −0.1999] |
| frac. pixels saturation > 0.8 | 0.3284 | 0.0297 | **−0.2987** | [−0.3137, −0.2838] |
| frac. clipped pixels | 0.1503 | 0.0045 | **−0.1458** | [−0.1534, −0.1383] |
| seed diversity (DINOv2, mean pairwise 1−cos) | 0.2951 | 0.3020 | +0.0069 | [−0.0057, +0.0200] |

- Saturation metrics separate E0 from REF by > 20σ, so they are good E1 reproduction criteria.
- **No significant diversity gain** of the official DFD result over the DMD2 start with this metric (5 seeds per
  prompt, identical noise across models), despite the paper's diversity claim. To be followed up (other feature
  spaces / more seeds) before drawing conclusions.

### E1a_seed1 trajectory, all 8 checkpoints (2026-09-27)

`scripts/eval/trajectory.py E1a_seed1` → `runs/eval/trajectory_E1a_seed1.{json,md}`. Paired bootstrap over prompts,
`*` = 95 % CI excludes 0. Step = training iteration (a generator update happens every 5 iterations).

| step | Quality | ΔQ vs E0 | ΔQ vs REF | Motion Sm. | Dyn. Deg. | sat mean | sat>0.8 % | Δ vs REF (pp) | clip % | div DINOv2 | Δdiv vs E0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.8479 | – | -0.0020 | 0.9828 | 0.820 | 0.572 | 32.8 | +29.9* | 15.03 | 0.2951 | – |
| 5 | 0.8501 | +0.0022 | +0.0002 | 0.9824 | 0.855 | 0.529 | 26.2 | +23.2* | 11.57 | 0.2931 | -0.0020 |
| 10 | 0.8498 | +0.0019 | -0.0001 | 0.9828 | 0.830 | 0.503 | 22.1 | +19.2* | 9.37 | 0.2983 | +0.0032 |
| 15 | 0.8490 | +0.0011 | -0.0009 | 0.9830 | 0.805 | 0.486 | 19.4 | +16.4* | 7.91 | 0.3015 | +0.0065* |
| 20 | 0.8501 | +0.0022 | +0.0002 | 0.9829 | 0.825 | 0.466 | 16.3 | +13.3* | 6.28 | 0.3043 | +0.0092* |
| 25 | 0.8492 | +0.0013 | -0.0007 | 0.9829 | 0.810 | 0.448 | 13.9 | +10.9* | 5.15 | 0.3052 | +0.0101* |
| 30 | 0.8487 | +0.0008 | -0.0012 | 0.9829 | 0.800 | 0.439 | 12.5 | +9.6* | 4.46 | 0.3077 | +0.0127* |
| 35 | 0.8507 | +0.0028 | +0.0008 | 0.9834 | 0.820 | 0.436 | 11.8 | +8.8* | 3.95 | 0.3078 | +0.0127* |
| 40 | 0.8500 | +0.0021 | +0.0001 | 0.9835 | 0.805 | 0.437 | 11.6 | +8.6* | 3.83 | 0.3093 | +0.0143* |
| REF | 0.8499 | – | – | 0.9838 | 0.860 | 0.360 | 3.0 | – | 0.45 | 0.3020 | – |

Findings
1. **Saturation falls monotonically but plateaus far above REF.** frac sat>0.8: 32.8 → 22.1 (10) → 16.3 (20) → 12.5 (30)
   → 11.6 % (40); steps 30–40 change < 1 pp. REF is 3.0 %; E1a stays significantly more saturated at every step
   (+8.6 pp at 40). Clipped pixels: 15.0 → 3.8 % (REF 0.45 %). Mean saturation 0.572 → 0.437 (REF 0.360).
2. **Diversity rises monotonically** and is significantly above E0 from step 15 on (+0.0143 at 40). At step 40 it
   is nominally above REF (0.3093 vs 0.3020, +0.0073 [−0.0038, +0.0180], n.s.). E1a gains diversity where REF showed
   no significant gain over E0.
3. **VBench Quality is flat** (0.8487–0.8507, all within noise of E0 and REF). Imaging quality closes the gap to REF
   (−0.0157* at 0 → −0.0053 n.s. at 40). Background consistency stays ≈ +0.008–0.010 above REF.
4. **No sign of over-training by step 40 in E1a**: no metric degrades; saturation/diversity are still improving
   slowly or have plateaued.
5. **Reproduction check (spec §3 E1): FAILED on saturation.** VBench is within noise, but the saturation gap to REF is
   large and significant, so E1a does not reproduce `ours_wan_t2v_vipe/0000040`. Per spec: stop and report.
   Candidate causes: (a) teacher-copy fake_score vs. the authors' trained critic (spec §5 risk 1, the deferred E1b
   warm-up tests this); (b) training-seed / data-order noise with only 8 generator updates (§5 risk 2; seeds 2–3);
   (c) per-GPU batch 1 × accum 8 on 2 GPUs vs. the authors' likely 8 GPUs × batch 2 (replacement granularity).

- **2026-09-27, user decision:** the re-implementation is sufficient. E1a_seed1 is the DFD baseline; the saturation gap to REF is attributed to different training data (see `runs/NOTES_paper.md`). Moving on to the DFD-revised work.

### 2026-09-28 — E2 / E2s (real data, Mixkit), light eval at it20 / it40

Paired bootstrap over prompts (B − A, * = 95 % CI excludes 0), same 800-video subset; `runs/eval/compare_E2_seed1_it0020.json`.

| model | Quality | Motion Sm. | Dyn. Deg. | sat mean | sat>0.8 % | clip % | div DINOv2 |
|---|---|---|---|---|---|---|---|
| E0 | 0.8479 | 0.9828 | 0.820 | 0.572 | 32.8 | 15.03 | 0.2951 |
| E1a it20 | 0.8501 | 0.9829 | 0.825 | 0.466 | 16.3 | 6.28 | 0.3043 |
| E1a it40 | 0.8500 | 0.9835 | 0.805 | 0.437 | 11.6 | 3.83 | 0.3093 |
| E2 it20 | 0.8529 | 0.9842 | 0.910 | 0.448 | 14.2 | 4.48 | 0.2931 |
| E2 it40 | 0.8534 | 0.9835 | 0.920 | 0.415 | 9.1 | 2.26 | 0.2908 |
| E2s it20 | 0.8492 | 0.9842 | 0.860 | 0.465 | 17.4 | 6.38 | 0.2885 |
| E2s it40 | 0.8469 | 0.9848 | 0.905 | 0.416 | 11.0 | 3.22 | 0.2962 |
| REF | 0.8499 | 0.9838 | 0.860 | 0.360 | 3.0 | 0.45 | 0.3020 |

| model − base | ΔQuality | ΔDyn. Deg. | Δsat>0.8 (pp) | Δdiv DINOv2 |
|---|---|---|---|---|
| E2 it20 − E1a it20 | +0.0028 | +0.085* | −2.1* | −0.0112* |
| E2 it40 − E1a it40 | +0.0034 | +0.115* | −2.5* | −0.0186* |
| E2 it40 − E0 | +0.0055* | +0.100* | −23.7* | −0.0043 |
| E2s it20 − E1a it20 | −0.0009 | +0.035 | +1.1* | −0.0158* |
| E2s it40 − E1a it40 | −0.0032 | +0.100* | −0.6* | −0.0131* |
| E2s it40 − E0 | −0.0010 | +0.085* | −21.8* | +0.0011 |
| E2s it40 − E2 it40 | −0.0066* | −0.015 | +1.9* | +0.0054 |

Reading (single training seed; CIs cover prompt sampling only, not training-seed noise; one diversity metric):
1. Real data reproduces the problem: DFD's diversity gain (E1a +0.014 over E0) is gone with Mixkit (E2 ≈ E0),
   while de-saturation still works (even slightly better than E1a) and VBench Quality does not drop.
2. Keeping the student input on ViPE (E2s) does not restore diversity (E2s − E2 n.s.), so the loss of the diversity
   gain is not explained by the student-input shift; it comes with the teacher-side replacement by real data.
   E2s is somewhat worse than E2 in Quality and saturation.
3. F2: E2 and E1d have the same delta_sq / Var(Δ) / Lipschitz scale; the diagnostics do not show a larger
   regularizer variance for real data under the coupled pairing. E2s (uncoupled pairs) has ~4× delta_sq and ~2.4×
   Var(Δ). In all runs Δ is ~zero-mean (mean share ~0.1 %).
4. E2s it40 eval: seed-3 generation crashed (SIGSEGV) in job 2222246 and was regenerated alone (job 2222436,
   `SEEDS=3`); other seeds' videos reused (deterministic sampler, same seeds).

### 2026-09-29 — E3 / E3s (CFG uncond branch at the generated point), light eval at it20 / it40

| model | Quality | Motion Sm. | Dyn. Deg. | sat mean | sat>0.8 % | clip % | div DINOv2 |
|---|---|---|---|---|---|---|---|
| E3 it20 | 0.8486 | 0.9832 | 0.845 | 0.520 | 22.9 | 8.16 | 0.2972 |
| E3 it40 | 0.8452 | 0.9838 | 0.825 | 0.415 | 9.3 | 3.04 | 0.2945 |
| E3s it20 | 0.8453 | 0.9868 | 0.770 | 0.560 | 27.9 | 10.23 | 0.2793 |
| E3s it40 | 0.8427 | 0.9887 | 0.725 | 0.507 | 19.5 | 6.35 | 0.2714 |

| model − base | ΔQuality | ΔMotion Sm. | ΔDyn. Deg. | Δsat>0.8 (pp) | Δclip (pp) | Δdiv DINOv2 |
|---|---|---|---|---|---|---|
| E3 it20 − E2 it20 | −0.0043* | −0.0009* | −0.065* | +8.7* | +3.7* | +0.0041 |
| E3 it40 − E2 it40 | −0.0082* | +0.0002 | −0.095* | +0.2 | +0.8* | +0.0038 |
| E3 it40 − E1a it40 | −0.0048* | +0.0002 | +0.020 | −2.3* | −0.8* | −0.0148* |
| E3 it40 − E0 | −0.0027 | +0.0009* | +0.005 | −23.5* | −12.0* | −0.0005 |
| E3s it20 − E2s it20 | −0.0039* | +0.0026* | −0.090* | +10.6* | +3.9* | −0.0092* |
| E3s it40 − E2s it40 | −0.0041 | +0.0039* | −0.180* | +8.5* | +3.1* | −0.0248* |
| E3s it40 − E0 | −0.0052* | +0.0059* | −0.095* | −13.3* | −8.7* | −0.0237* |

Reading (single seed, prompt-level CIs only):
1. E3 does not recover DFD's diversity gain (E3 ≈ E0 ≈ E2 on diversity) and costs Quality and the Mixkit motion gain
   relative to E2; de-saturation reaches E2's level only by it40 (slower early).
2. On the E2s arm the CFG variant is clearly harmful: diversity falls below the E0 start, motion drops (Dyn. Deg.
   0.725, higher Motion Smoothness = less motion), saturation is reduced much less. With uncoupled (x, G) pairs
   (E2s delta_sq ≈ 4× E2) the guidance term (w−1)(x0_cond(x_t^real) − x0_uncond(x_t^gen)) mixes two unrelated inputs,
   and the student appears to collapse toward a low-motion mode.
3. F2 (standard-CFG Δ, `runs/eval/diag_E1d_E2_E3.md`): E3 ≈ E2 (delta_sq 0.357, Var 3.31e5), E3s ≈ E2s (1.209, 9.27e5);
   by construction these diagnostics do not measure the E3 target itself.
