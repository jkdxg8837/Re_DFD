# SPDX-License-Identifier: Apache-2.0
"""Pixel statistics of the s_score clips (user task data_analysis_task.md, qualitative check made quantitative):
per clip, over frames 0..80 (every 4th) after the training transform (resize small side, center crop 832x480):
mean luma, fraction of dark (luma < 0.1) / bright (luma > 0.9) pixels, mean HSV saturation; then the within-dataset
Spearman correlation with s_neg / s_emp (t = 0.999). Reads runs/s_score/n256/s_scores.csv; CPU only.

  python scripts/diag/s_score_pixstats.py --out runs/s_score/n256 [--procs 8]
"""

from __future__ import annotations

import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np

W, H, SEQ = 832, 480, 81
FRAMES = list(range(0, SEQ, 4))


def stats(args):
    ds, n, want = args
    from PIL import Image

    from scripts.diag.delta_scale import decode
    from scripts.diag.s_score import sources

    out = {}
    for cid, mp4, _ in sources(ds, n):
        if cid not in want:
            continue
        v = decode(mp4)[:SEQ]
        lum, dark, bright, sat = [], [], [], []
        for fi in FRAMES:
            im = Image.fromarray(v[fi])
            s = max(W / im.width, H / im.height)
            im = im.resize((max(W, round(im.width * s)), max(H, round(im.height * s))), Image.BILINEAR)
            l, tp = (im.width - W) // 2, (im.height - H) // 2
            im = im.crop((l, tp, l + W, tp + H))
            a = np.asarray(im, dtype=np.float32) / 255
            y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
            mx, mn = a.max(-1), a.min(-1)
            lum.append(y.mean())
            dark.append((y < 0.1).mean())
            bright.append((y > 0.9).mean())
            sat.append(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())
        out[cid] = dict(luma=float(np.mean(lum)), dark_frac=float(np.mean(dark)), bright_frac=float(np.mean(bright)),
                        hsv_sat=float(np.mean(sat)))
    return ds, out


def main():
    from scipy.stats import spearmanr

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(args.out, "s_scores.csv"))))
    dss = ["vipe", "mixkit", "openvid"]
    # one task per (dataset, chunk of clip ids); sources() streams the dataset, chunks only filter
    tasks = []
    for ds in dss:
        ids = [r["vid"] for r in rows if r["ds"] == ds]
        n = len(ids)
        k = max(1, args.procs // len(dss) + 1)
        for c in range(k):
            tasks.append((ds, n, set(ids[c::k])))
    px = {ds: {} for ds in dss}
    with Pool(args.procs) as p:
        for ds, o in p.imap_unordered(stats, tasks):
            px[ds].update(o)
    feats = ["luma", "dark_frac", "bright_frac", "hsv_sat"]
    with open(os.path.join(args.out, "pixstats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ds", "vid", *feats])
        for ds in dss:
            for r in rows:
                if r["ds"] == ds:
                    w.writerow([ds, r["vid"], *[px[ds][r["vid"]][k] for k in feats]])
    lines = ["### Pixel statistics vs s (t = 0.999; frames 0..80 every 4th, training crop)", "",
             "Spearman rho (p) within dataset.", "",
             "| dataset | score | " + " | ".join(feats) + " |", "|---|---|" + "---|" * len(feats)]
    for ds in dss:
        R = [r for r in rows if r["ds"] == ds]
        for sc in ["s_neg", "s_emp"]:
            cells = []
            for k in feats:
                rr, pp = spearmanr([float(r[sc]) for r in R], [px[ds][r["vid"]][k] for r in R])
                cells.append(f"{rr:+.3f} ({pp:.1g})")
            lines.append(f"| {ds} | {sc} | " + " | ".join(cells) + " |")
    lines += ["", "Dataset means: " + "; ".join(
        f"{ds} " + ", ".join(f"{k} {np.mean([px[ds][v][k] for v in px[ds]]):.3f}" for k in feats) for ds in dss)]
    # top vs bottom third by s_neg within dataset
    lines += ["", "| dataset | third by s_neg | n | " + " | ".join(feats) + " |", "|---|---|---|" + "---|" * len(feats)]
    for ds in dss:
        R = sorted([r for r in rows if r["ds"] == ds], key=lambda r: float(r["s_neg"]))
        k = len(R) // 3
        for name, part in [("low", R[:k]), ("mid", R[k:-k]), ("high", R[-k:])]:
            lines.append(f"| {ds} | {name} | {len(part)} | " + " | ".join(
                f"{np.mean([px[ds][r['vid']][f] for r in part]):.3f}" for f in feats) + " |")
    open(os.path.join(args.out, "pixstats.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
