# E3s_seed1: E2s + CFG variant (spec §3 E3 on the second Mixkit arm)

Change vs E2s_seed1: `++model.dfd_cfg_uncond_on_gen=True`; student input = paired ViPE `long` video, replacement =
Mixkit, all Mixkit captions (data `data/mixkit_vipe_pair_wds_v3`). Same hypothesis as E3; compare E3s − E2s with
E3 − E2 to see whether the CFG change interacts with the student-input source.

## Conclusion (2026-09-29)
Train 2223446 (1 h 47 min), evals 2223447/48. it40 vs E2s it40: diversity −0.0248* (0.2714, below E0), Dyn. Deg.
−0.180* (0.725), sat>0.8 +8.5 pp* (19.5 %), Motion Smoothness +0.004* → clearly harmful; mode-seeking toward
low-motion outputs. The CFG change interacts with the uncoupled pairing of the E2s arm (E3 − E2 was mostly neutral
on diversity, E3s − E2s is strongly negative).
