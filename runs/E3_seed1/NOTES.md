# E3_seed1: DFD + real data (Mixkit) + CFG variant, spec §3 E3

Change vs E2_seed1: `++model.dfd_cfg_uncond_on_gen=True` [CONFIRMED 2026-09-28]. On replaced batches (p=0.5, whole
micro-batch) the teacher target is x0 = x0_cond(x_t^real, c) + (w−1)(x0_cond(x_t^real, c) − x0_uncond(x_t^gen)),
w=5; eps and t still shared; non-replaced batches unchanged. Data `data/mixkit_wds_v3`, seed 1, 40 steps, F2 on.

Hypothesis (before the run): the CFG "direction" term then contains real-vs-gen displacement at the conditional
input only, so the unconditional anchor stays on the student's own distribution; if E2's loss of the diversity
gain comes from the real point pulling both branches (cond and uncond) toward the real-data mode, E3 should recover
part of E1a's diversity gain (div above E2 it40 = 0.2908) while keeping de-saturation. Risk: the guidance term
(w−1)(x0_cond(real) − x0_uncond(gen)) mixes two different inputs and can be large (4×), which may over-sharpen or
destabilize (watch saturation / clip %).
F2 note: with the flag on, the diagnostics recompute the standard CFG prediction at both points (Δ keeps the E2
definition, i.e. it is not the E3 target).

## Conclusion (2026-09-29)
Train 2223443 (1 h 36 min), evals 2223444/45. it40 vs E2 it40: diversity +0.004 n.s. (0.2945 ≈ E0), Quality −0.0082*,
Dyn. Deg. −0.095*, sat>0.8 +0.2 pp n.s.; at it20 de-saturation is slower (22.9 % vs 14.2 %). → **Hypothesis not
supported**: keeping the uncond anchor on the generated point does not recover the diversity gain and costs quality
and motion. F2 (standard-CFG Δ) ≈ E2. vsd_loss is higher in the last student updates (0.83/0.82/0.71 vs E2 0.62/0.81/0.37).
