# SPDX-License-Identifier: Apache-2.0
"""Collect per-step metrics of one run and compare every step with E0 and REF (paired bootstrap over prompts).

Usage: python scripts/eval/trajectory.py <run_prefix> [--steps 5 10 ...] [--e0 E0_dmd2_T21] [--ref REF_ours_T21]
       e.g. python scripts/eval/trajectory.py E1a_seed1
Writes runs/eval/trajectory_<run_prefix>.json and prints a Markdown table.
"""

import argparse
import json
import os
import subprocess
import sys

EXTRA = ["sat_mean", "frac_sat_gt_08", "clip_frac", "div_dino"]


def boot_vbench(a, b):
    out = f"/tmp/_boot_{os.getpid()}.json"
    subprocess.run([sys.executable, "scripts/eval/paired_bootstrap.py", f"runs/eval/{a}/vbench/{a}_eval_results.json",
                    f"runs/eval/{b}/vbench/{b}_eval_results.json", "--out", out], check=True, capture_output=True)
    r = json.load(open(out))["rows"]
    os.remove(out)
    return r


def boot_extra(a, b):
    subprocess.run([sys.executable, "scripts/eval/compare_extra.py", a, b], check=True, capture_output=True)
    return json.load(open(f"runs/eval/extra_{a}_vs_{b}.json"))["rows"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--steps", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30, 35, 40])
    ap.add_argument("--e0", default="E0_dmd2_T21")
    ap.add_argument("--ref", default="REF_ours_T21")
    a = ap.parse_args()
    tags = [(0, a.e0)] + [(s, f"{a.run}_it{s:04d}") for s in a.steps]
    rows = []
    for step, tag in tags:
        q = json.load(open(f"runs/eval/{tag}/vbench/quality_score.json"))
        ex = json.load(open(f"runs/eval/{tag}/extra/extra_metrics.json"))["summary"]
        row = {"step": step, "tag": tag, "quality": q["quality_score"], **{k: v for k, v in q["raw"].items()},
               **{k: ex[k] for k in EXTRA}}
        for base in (a.e0, a.ref):
            if base == tag:
                continue
            vb, xb = boot_vbench(base, tag), boot_extra(base, tag)
            key = "vsE0" if base == a.e0 else "vsREF"
            row[key] = {"quality": vb["quality_score"], **{d: vb[d] for d in ("motion_smoothness", "dynamic_degree",
                        "imaging_quality", "aesthetic_quality", "background_consistency")},
                        **{m: xb[m] for m in EXTRA}}
        rows.append(row)
        print(f"done {tag}", file=sys.stderr, flush=True)
    refq = json.load(open(f"runs/eval/{a.ref}/vbench/quality_score.json"))
    refx = json.load(open(f"runs/eval/{a.ref}/extra/extra_metrics.json"))["summary"]
    rows.append({"step": "REF", "tag": a.ref, "quality": refq["quality_score"], **refq["raw"], **{k: refx[k] for k in EXTRA}})
    json.dump(rows, open(f"runs/eval/trajectory_{a.run}.json", "w"), indent=1)

    def ci(r, key, m, pct=False):
        if key not in r:
            return "–"
        c = r[key][m]
        f = (lambda x: f"{100 * x:+.1f}") if pct else (lambda x: f"{x:+.4f}")
        sig = "*" if c["ci95"][0] > 0 or c["ci95"][1] < 0 else ""
        return f"{f(c['diff'])}{sig}"

    print("| step | Quality | ΔQ vs E0 | ΔQ vs REF | Motion Sm. | Dyn. Deg. | sat mean | sat>0.8 % | Δ vs REF (pp) | clip % | div DINOv2 | Δdiv vs E0 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['step']} | {r['quality']:.4f} | {ci(r, 'vsE0', 'quality')} | {ci(r, 'vsREF', 'quality')} | "
              f"{r['motion_smoothness']:.4f} | {r['dynamic_degree']:.3f} | {r['sat_mean']:.3f} | "
              f"{100 * r['frac_sat_gt_08']:.1f} | {ci(r, 'vsREF', 'frac_sat_gt_08', pct=True)} | {100 * r['clip_frac']:.2f} | "
              f"{r['div_dino']:.4f} | {ci(r, 'vsE0', 'div_dino')} |")


if __name__ == "__main__":
    main()
