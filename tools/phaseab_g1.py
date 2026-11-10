# SPDX-License-Identifier: Apache-2.0
"""Assemble the Gate G1 table (task2.md) from runs/phaseAB/metrics/*.json -> runs/phaseAB/G1_table.md (+ json)."""

import json
import os

ROOT = os.environ.get("PHASEAB_ROOT", "runs/phaseAB")  # data_prep.md A-1..A-3 redo: runs/phaseAB/v2
M = f"{ROOT}/metrics"
# supplementary A1_w1 sigma: read from its own blur fit (round 1: 3.0, v2: 3.6); round-1 default kept
_bf = f"{ROOT}/blur/blurfit_A1_w1.json"
SIG_W1 = json.load(open(_bf))["best_sigma"] if os.path.exists(_bf) else 3
ROWS = [("S_vipe", "S_vipe", ""), ("R_mx", "R_mx", ""), ("R_ov", "R_ov", ""), ("T_w1", "T_w1", ""),
        ("T_w5", "T_w5", ""), ("A1 w=5", "A1_w5", "_blur0.5"), ("A1 w=1", "A1_w1", "_blur0.5"),
        ("A1 w=1 (σ=%g refs)" % SIG_W1, "A1_w1", "_blur%g" % SIG_W1), ("dmd2", "dmd2", ""), ("dfd40", "dfd40", ""), ("E1a", "E1a", ""),
        ("E2", "E2", "")]


def f(d, k, p=3):
    v = d.get(k)
    if not isinstance(v, dict):
        return "–"
    s = f"{v['mean']:.{p}f}"
    if "ci" in v:
        s += f" [{v['ci'][0]:.{p}f}, {v['ci'][1]:.{p}f}]"
    return s


lines = ["| set | π [95% CI] | π_mx | π_ov | probe logit (b) | real frac (b) | probe logit (a) | luminance | saturation |"
         " dark ratio | FD R_mx | FD R_ov | FD S_vipe | prec/rec R_ov |",
         "|" + "---|" * 14]
for label, name, suf in ROWS:
    p = os.path.join(M, f"{name}{suf}.json")
    if not os.path.exists(p):
        lines.append(f"| {label} | (missing) |")
        continue
    d = json.load(open(p))
    pr = d.get("knn_precision_R_ov", {}).get("mean"), d.get("knn_recall_R_ov", {}).get("mean")
    lines.append(f"| {label}{' (vs refs' + suf + ')' if suf else ''} | {f(d, 'pi')} | {f(d, 'pi_mx', 2)} | "
                 f"{f(d, 'pi_ov', 2)} | {f(d, 'probe_logit_b', 1)} | {d['probe_real_frac_b']['mean']:.2f} | "
                 f"{f(d, 'probe_logit_a', 1)} | {f(d, 'luminance')} | {f(d, 'saturation')} | {f(d, 'dark_ratio')} | "
                 f"{d['FD_R_mx']['mean']:.0f} | {d['FD_R_ov']['mean']:.0f} | {d['FD_S_vipe']['mean']:.0f} | "
                 f"{pr[0]:.2f}/{pr[1]:.2f} |")
    if suf:
        base = " / ".join(f"{r} {d[f'ref{suf}_probe_logit_b_{r}']['mean']:.1f}" for r in ["R_mx", "R_ov", "S_vipe"])
        lines.append(f"| ↳ probe logit (b) of the low-passed refs{suf} | {base} |" + " |" * 12)
by = []
for name in ["T_w5", "T_w1", "A1_w5", "A1_w1", "dmd2", "dfd40", "E1a", "E2"]:
    p = os.path.join(M, f"{name}.json")
    if os.path.exists(p) and "by_source" in json.load(open(p)):
        bs = json.load(open(p))["by_source"]
        by.append(f"| {name} | " + " | ".join(
            f"{bs[s]['pi']['mean']:.3f} / {bs[s]['probe_logit_b']['mean']:.1f} / {bs[s]['saturation']['mean']:.3f}"
            for s in ["vipe", "mixkit", "openvid"]) + " |")
lines += ["", "π / probe logit (b) / saturation by prompt source:", "", "| set | ViPE prompts | Mixkit prompts |"
          " OpenVid prompts |", "|---|---|---|---|"] + by
open(f"{ROOT}/G1_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
