# E2_seed1: DFD + real-world data (Mixkit), spec §3 E2

Change vs E1a_seed1: `dataloader_train.datatags` → `data/mixkit_wds` (Mixkit, Qwen2.5-VL-7B v2 captions); F2
diagnostics on (no effect on the loss, `tests/test_dfd_flags.py`). Everything else identical (seed 1, 40 steps,
GAN off, key map {net: net}, T=21, 2×A100 bs1×accum8).

Hypothesis (before the run): per the Notion problem statement, real data gives no gain or is worse than E1a
(saturation reduction smaller than E1a's 32.8 → 11.6 % and/or quality drops).
§1.1 side effect: the data enters twice. (1) Teacher-score replacement (p=0.5, whole batch). (2) The multi-step
student input is forward_process(x_mixkit, eps, t_student): for t_student ∈ {0.937, 0.833, 0.624} the student
sees 6–38 % Mixkit signal, so its inputs move away from the ViPE-like distribution it was distilled on, and
‖x − G‖ is small by construction (G is predicted from a noised copy of x). E2s isolates (1).
Eval (light, spec §2.4): checkpoints 20 and 40, 800-video VBench subset + saturation + DINOv2 diversity, paired with E1a.

## Conclusion (2026-09-28)
Train job 2222241 (1 h 32 min), 8 student + 32 critic updates, no errors; evals 2222242/2222243.
it40 vs E1a it40: diversity (DINOv2) −0.0186* (E2 ≈ E0: −0.004 n.s.), sat>0.8 −2.5 pp* (9.1 % vs 11.6 %),
Dynamic Degree +0.115*, Quality +0.003 n.s. → **hypothesis partly confirmed**: real data kills DFD's diversity gain
but not its de-saturation, and does not hurt quality. F2 (x0 space): delta_sq 0.309, Var(Δ) 3.36e5 — same scale as
E1d (0.358, 3.70e5); no larger regularizer variance for real data under the coupled pairing. Single seed.
