# SPDX-License-Identifier: Apache-2.0
"""Paired bootstrap of VBench subset results for two models evaluated on the same prompts and seeds.

Resampling unit = prompt (its 5 seed videos stay together). The dimensions of one official prompt group share the
same resampled prompts; groups are resampled independently. Each replicate recomputes every dimension mean and the
Quality Score (official normalization / weights) for both models; the script reports B − A with a 95 % percentile CI.

Usage: python scripts/eval/paired_bootstrap.py <A_eval_results.json> <B_eval_results.json> [--n 10000] [--out f.json]
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vbench_quality_score import NORMALIZE, WEIGHT  # noqa: E402

SCALE = {"imaging_quality": 0.01}  # per-video MUSIQ scores are 0-100, the aggregate is 0-1


def per_video(results: dict) -> dict:
    """dim -> {(prompt, k): value}"""
    out = {}
    for d in NORMALIZE:
        vals = {}
        for v in results[d][1]:
            m = re.match(r"(.*)-(\d+)\.mp4$", os.path.basename(v["video_path"]))
            vals[(m.group(1), int(m.group(2)))] = float(v["video_results"]) * SCALE.get(d, 1.0)
        out[d] = vals
    return out


def quality(means: dict) -> float:
    return sum(WEIGHT[d] * (means[d] - lo) / (hi - lo) for d, (lo, hi) in NORMALIZE.items()) / sum(WEIGHT.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A, B = per_video(json.load(open(args.a))), per_video(json.load(open(args.b)))

    # prompt groups: dims with identical prompt sets
    groups = {}
    for d in NORMALIZE:
        assert A[d].keys() == B[d].keys(), f"{d}: videos differ between A and B"
        prompts = tuple(sorted({p for p, _ in A[d]}))
        groups.setdefault(prompts, []).append(d)
    # per dim: matrix [n_prompts, 5] for A and B, rows ordered like the group's prompt tuple
    mats = {}
    for prompts, dims in groups.items():
        for d in dims:
            ks = sorted({k for _, k in A[d]})
            mats[d] = tuple(np.array([[M[d][(p, k)] for k in ks] for p in prompts]) for M in (A, B))

    rng = np.random.default_rng(args.seed)
    boot = {d: np.empty((args.n, 2)) for d in NORMALIZE}
    for prompts, dims in groups.items():
        idx = rng.integers(0, len(prompts), size=(args.n, len(prompts)))
        for d in dims:
            ma, mb = mats[d]
            boot[d][:, 0] = ma.mean(1)[idx].mean(1)
            boot[d][:, 1] = mb.mean(1)[idx].mean(1)
    qa = np.array([quality({d: boot[d][i, 0] for d in NORMALIZE}) for i in range(args.n)])
    qb = np.array([quality({d: boot[d][i, 1] for d in NORMALIZE}) for i in range(args.n)])

    rows = {}
    point = {d: (mats[d][0].mean(), mats[d][1].mean()) for d in NORMALIZE}
    for d in NORMALIZE:
        diff = boot[d][:, 1] - boot[d][:, 0]
        rows[d] = {"A": point[d][0], "B": point[d][1], "diff": point[d][1] - point[d][0],
                   "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                   "p_B_gt_A": float((diff > 0).mean())}
    qd = qb - qa
    q_point = (quality({d: point[d][0] for d in NORMALIZE}), quality({d: point[d][1] for d in NORMALIZE}))
    rows["quality_score"] = {"A": q_point[0], "B": q_point[1], "diff": q_point[1] - q_point[0],
                             "ci95": [float(np.percentile(qd, 2.5)), float(np.percentile(qd, 97.5))],
                             "p_B_gt_A": float((qd > 0).mean())}
    print(f"A = {args.a}\nB = {args.b}\nbootstrap n={args.n}, unit=prompt, groups={list(groups.values())}")
    print(f"{'metric':24s} {'A':>8s} {'B':>8s} {'B-A':>8s}   95% CI of B-A        P(B>A)")
    for k, r in rows.items():
        sig = "*" if r["ci95"][0] > 0 or r["ci95"][1] < 0 else " "
        print(f"{k:24s} {r['A']:8.4f} {r['B']:8.4f} {r['diff']:+8.4f}   [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] {sig}  "
              f"{r['p_B_gt_A']:.3f}")
    if args.out:
        json.dump({"a": args.a, "b": args.b, "n": args.n, "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
