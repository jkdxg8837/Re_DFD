# SPDX-License-Identifier: Apache-2.0
"""Paired bootstrap (unit = prompt) of the extra metrics between two tags: reports B − A with a 95 % CI.

Usage: python scripts/eval/compare_extra.py <tag_A> <tag_B> [--n 10000] [--eval_root runs/eval]
"""

import argparse
import json
import os

import numpy as np

METRICS = ["sat_mean", "frac_sat_gt_08", "clip_frac", "div_dino"]


def per_prompt_table(path: str) -> dict:
    d = json.load(open(path))
    tab = {p: {"div_dino": v["div_dino"]} for p, v in d["per_prompt"].items()}
    for v in d["per_video"].values():
        for m in ("sat_mean", "frac_sat_gt_08", "clip_frac"):
            tab[v["prompt"]].setdefault(m, []).append(v[m])
    return {p: {m: float(np.mean(x)) for m, x in r.items()} for p, r in tab.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--eval_root", default="runs/eval")
    args = ap.parse_args()
    A = per_prompt_table(os.path.join(args.eval_root, args.a, "extra", "extra_metrics.json"))
    B = per_prompt_table(os.path.join(args.eval_root, args.b, "extra", "extra_metrics.json"))
    prompts = sorted(A)
    assert prompts == sorted(B), "prompt sets differ"
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(prompts), size=(args.n, len(prompts)))
    rows = {}
    print(f"A={args.a}  B={args.b}  (paired bootstrap over {len(prompts)} prompts, n={args.n})")
    print(f"{'metric':16s} {'A':>8s} {'B':>8s} {'B-A':>9s}   95% CI of B-A")
    for m in METRICS:
        a = np.array([A[p][m] for p in prompts])
        b = np.array([B[p][m] for p in prompts])
        diff = (b - a)[idx].mean(1)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows[m] = {"A": a.mean(), "B": b.mean(), "diff": b.mean() - a.mean(), "ci95": [lo, hi]}
        sig = "*" if lo > 0 or hi < 0 else " "
        print(f"{m:16s} {a.mean():8.4f} {b.mean():8.4f} {b.mean() - a.mean():+9.4f}   [{lo:+.4f}, {hi:+.4f}] {sig}")
    out = os.path.join(args.eval_root, f"extra_{args.a}_vs_{args.b}.json")
    json.dump({"a": args.a, "b": args.b, "n": args.n, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
