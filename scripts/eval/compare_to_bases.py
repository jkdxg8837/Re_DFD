# SPDX-License-Identifier: Apache-2.0
"""Light-eval comparison: each tag vs a list of base tags (paired bootstrap over prompts, B - A, 95 % CI).

Usage: python scripts/eval/compare_to_bases.py --tags E2_seed1_it0020 ... --bases E0_dmd2_T21 REF_ours_T21 E1a_seed1_it{step}
       ("{step}" in a base is replaced by the tag's 4-digit step, e.g. E1a_seed1_it0020)
Prints a Markdown table; writes runs/eval/compare_<first tag>.json.
"""

import argparse
import json
import os
import re
import subprocess
import sys

EXTRA = ["sat_mean", "frac_sat_gt_08", "clip_frac", "div_dino"]
VB = ["quality_score", "motion_smoothness", "dynamic_degree", "imaging_quality", "aesthetic_quality"]


def boot_vbench(a, b):
    out = f"/tmp/_cmp_{os.getpid()}.json"
    subprocess.run([sys.executable, "scripts/eval/paired_bootstrap.py", f"runs/eval/{a}/vbench/{a}_eval_results.json",
                    f"runs/eval/{b}/vbench/{b}_eval_results.json", "--out", out], check=True, capture_output=True)
    r = json.load(open(out))["rows"]
    os.remove(out)
    return r


def boot_extra(a, b):
    subprocess.run([sys.executable, "scripts/eval/compare_extra.py", a, b], check=True, capture_output=True)
    return json.load(open(f"runs/eval/extra_{a}_vs_{b}.json"))["rows"]


def fmt(c, pct=False):
    f = (lambda x: f"{100 * x:+.1f}") if pct else (lambda x: f"{x:+.4f}")
    sig = "*" if c["ci95"][0] > 0 or c["ci95"][1] < 0 else ""
    return f"{f(c['diff'])}{sig}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--bases", nargs="+", default=["E0_dmd2_T21", "REF_ours_T21", "E1a_seed1_it{step}"])
    a = ap.parse_args()
    res = {}
    print("| model | Quality | Motion Sm. | Dyn. Deg. | sat mean | sat>0.8 % | clip % | div DINOv2 |")
    print("|---|---|---|---|---|---|---|---|")
    rows = {}
    for t in a.tags + sorted({b.replace("{step}", re.search(r"it(\d{4})", t)[1]) for t in a.tags for b in a.bases}):
        q = json.load(open(f"runs/eval/{t}/vbench/quality_score.json"))
        ex = json.load(open(f"runs/eval/{t}/extra/extra_metrics.json"))["summary"]
        rows[t] = {"quality": q["quality_score"], **q["raw"], **{k: ex[k] for k in EXTRA}}
        r = rows[t]
        print(f"| {t} | {r['quality']:.4f} | {r['motion_smoothness']:.4f} | {r['dynamic_degree']:.3f} | {r['sat_mean']:.3f} | "
              f"{100 * r['frac_sat_gt_08']:.1f} | {100 * r['clip_frac']:.2f} | {r['div_dino']:.4f} |")
    print()
    print("| model − base | ΔQuality | ΔMotion Sm. | ΔDyn. Deg. | Δsat>0.8 (pp) | Δclip (pp) | Δdiv DINOv2 |")
    print("|---|---|---|---|---|---|---|")
    for t in a.tags:
        step = re.search(r"it(\d{4})", t)[1]
        for b in a.bases:
            b = b.replace("{step}", step)
            vb, xb = boot_vbench(b, t), boot_extra(b, t)
            res[f"{t}_vs_{b}"] = {"vbench": {d: vb[d] for d in VB}, "extra": {m: xb[m] for m in EXTRA}}
            print(f"| {t} − {b} | {fmt(vb['quality_score'])} | {fmt(vb['motion_smoothness'])} | {fmt(vb['dynamic_degree'])} | "
                  f"{fmt(xb['frac_sat_gt_08'], True)} | {fmt(xb['clip_frac'], True)} | {fmt(xb['div_dino'])} |")
    json.dump({"rows": rows, "compare": res}, open(f"runs/eval/compare_{a.tags[0]}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
