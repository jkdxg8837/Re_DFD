# SPDX-License-Identifier: Apache-2.0
"""Summarize F2 diagnostics (spec §4, first three metrics) written by model.dfd_diagnostics_every.

Usage: python scripts/eval/diag_summary.py <diag_dir> [<diag_dir> ...] [--out runs/eval/diag_<name>.md]
Per diag dir (one run), per student step and pooled over the run, per t bin:
  reg_mean_norm = ||E[Delta]||^2, unbiased pairwise estimate from count sketches:
                  (||sum_i S Delta_i||^2 - sum_i ||S Delta_i||^2) / (n (n - 1))
  reg_var       = E||Delta||^2 (exact norms) - reg_mean_norm
  mean_frac     = reg_mean_norm / E||Delta||^2 (share of the regularizer's energy that is mean, not variance)
  delta_sq      = E mean_elem (x_real - G)^2 (latent)
  lipschitz_hat = E ||Delta|| / ||x_t^real - x_t^gen||
Delta is reported in score space and in x0 space (Delta * sigma^2 / alpha = teacher-x0 difference minus (x - G)),
since score-space norms scale like 1/sigma^4 and are dominated by the smallest t.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def load(diag_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(diag_dir, "rank*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    sk = {}
    for f in glob.glob(os.path.join(diag_dir, "sketch_*.pt")):
        tag = os.path.basename(f)[len("sketch_"):-3]
        sk[tag] = torch.load(f)["sketch"].double().numpy()
    for r in rows:
        r["S"] = sk[r["tag"]][r["b"]]
    return rows


def stats(rows, space):
    n = len(rows)
    if n == 0:
        return None
    c = np.array([1.0 if space == "score" else r["sigma"] ** 2 / r["alpha"] for r in rows])
    S = np.stack([r["S"] for r in rows]) * c[:, None]
    sq = np.array([r["score_diff_sq"] for r in rows]) * c**2
    e_sq = sq.mean()
    if n > 1:
        tot = S.sum(0)
        mean_norm = (tot @ tot - (S * S).sum()) / (n * (n - 1))
    else:
        mean_norm = float("nan")
    return {"n": n, "reg_mean_norm": float(mean_norm), "reg_var": float(e_sq - mean_norm), "E_sq": float(e_sq),
            "mean_frac": float(mean_norm / e_sq) if e_sq > 0 else float("nan")}


def summarize(rows):
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r["student_step"]].append(r)
    groups["all"] = rows
    for g, rs in groups.items():
        per_bin = {}
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            b = [r for r in rs if lo <= r["t"] < hi or (hi == 1.0 and r["t"] == 1.0)]
            per_bin[f"[{lo:.1f},{hi:.1f})"] = {
                "score": stats(b, "score"), "x0": stats(b, "x0"),
                "delta_sq": float(np.mean([r["delta_sq"] for r in b])) if b else None,
                "lipschitz_hat": float(np.mean([r["lipschitz_hat"] for r in b])) if b else None,
            }
        out[str(g)] = {"n": len(rs), "x0_all_t": stats(rs, "x0"),
                       "delta_sq": float(np.mean([r["delta_sq"] for r in rs])),
                       "lipschitz_hat": float(np.mean([r["lipschitz_hat"] for r in rs])),
                       "frac_replaced": float(np.mean([r["replaced"] for r in rs])), "bins": per_bin}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {d: summarize(load(d)) for d in a.dirs}
    lines = ["| run | student step | n | delta_sq | lipschitz_hat | x0: ||E Delta||^2 | x0: Var | x0: mean share |",
             "|---|---|---|---|---|---|---|---|"]
    for d, s in res.items():
        keys = sorted([k for k in s if k != "all"], key=int) + ["all"]
        for k in keys:
            v = s[k]
            x = v["x0_all_t"]
            lines.append(f"| {os.path.basename(os.path.dirname(d.rstrip('/')))} | {k} | {v['n']} | {v['delta_sq']:.4f} | "
                         f"{v['lipschitz_hat']:.3f} | {x['reg_mean_norm']:.3e} | {x['reg_var']:.3e} | {x['mean_frac']:.3f} |")
    lines += ["", "Per t bin (pooled over the run, x0 space):", "",
              "| run | t bin | n | delta_sq | lipschitz_hat | ||E Delta||^2 | Var | mean share |", "|---|---|---|---|---|---|---|---|"]
    for d, s in res.items():
        for b, v in s["all"]["bins"].items():
            x = v["x0"]
            if x is None:
                continue
            lines.append(f"| {os.path.basename(os.path.dirname(d.rstrip('/')))} | {b} | {x['n']} | {v['delta_sq']:.4f} | "
                         f"{v['lipschitz_hat']:.3f} | {x['reg_mean_norm']:.3e} | {x['reg_var']:.3e} | {x['mean_frac']:.3f} |")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        json.dump(res, open(a.out.replace(".md", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
