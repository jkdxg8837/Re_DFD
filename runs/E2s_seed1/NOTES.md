# E2s_seed1: DFD + real data on the teacher side only, spec §3 E2 (second arm)

Change vs E2_seed1: the multi-step student input is built from a paired ViPE `long` video
(`++model.dfd_student_input_key=student_real`, data `data/mixkit_vipe_pair_wds`); the teacher-score replacement still
uses the Mixkit video. Student, fake score and teacher all use the Mixkit caption. Each Mixkit sample is paired
with the ViPE video whose caption is the nearest neighbor (mpnet cosine), so the noised ViPE input roughly
matches the caption. The student input uses a fresh eps' (same t_student).

Hypothesis (before the run): if E2's failure comes from the student-input shift (§1.1 point 2), E2s behaves like E1a;
if it comes from the replacement itself (real points far from G, loose A.1/A.2 constants), E2s behaves like E2.
Note: with ViPE input and a Mixkit caption, the pair is not coupled as in E1/E2 (G is predicted from a different
video than x), so delta_sq is expected to be larger than in E2. Compare F2 diagnostics E1d / E2 / E2s.

## Conclusion (2026-09-28)
Train job 2222244 (1 h 43 min), no errors; evals 2222245 (it20), 2222246 → seed 3 SIGSEGV, regenerated in 2222436.
it40 vs E1a it40: diversity −0.0131* (≈ E0, +0.001), sat>0.8 −0.6 pp*; vs E2 it40: diversity +0.005 n.s.,
Quality −0.0066*, sat +1.9 pp*. → Keeping the student input on synthetic data does **not** restore the diversity
gain, so the failure is tied to the teacher-side replacement with real data, not the student-input shift.
F2: delta_sq 1.269 (~4× E2), Var(Δ) 8.97e5 (~2.4×), as expected for uncoupled (x, G) pairs; the larger regularizer
variance does not translate into a different diversity outcome than E2. Single seed.
