"""Phase 0 item 6: contact sheets + simple saturation / diversity statistics for dmd2 vs ours samples.
Low-memory (no torch): frames are decoded by ffmpeg at 208x120."""
import glob
import json
import subprocess

import numpy as np
from PIL import Image, ImageDraw

FF = glob.glob(".venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-*")[0]
W, H = 208, 120
S = "runs/phase0/samples/{m}/seed{s}/prompts_first4/student_step4_{i:04d}_seed{s}.mp4"
MODELS = {"dmd2": "dmd2_t2v_vipe", "ours": "ours_wan_t2v_vipe"}


def load(path):
    raw = subprocess.run([FF, "-loglevel", "error", "-i", path, "-vf", f"scale={W}:{H}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def stats(v):
    x = v.astype(np.float32) / 255.0
    mx, mn = x.max(-1), x.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)          # HSV saturation
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return {"sat_mean": float(sat.mean()), "sat_p95": float(np.percentile(sat, 95)),
            "frac_sat_gt_0.8": float((sat > 0.8).mean()), "colorfulness": float(colorful),
            "brightness": float(x.mean()), "clip_frac": float(((x < 0.02) | (x > 0.98)).mean()),
            "temporal_diff": float(np.abs(np.diff(x, axis=0)).mean())}


rows, per_model = [], {k: [] for k in MODELS}
for i in range(4):
    vids = {(k, s): load(S.format(m=m, s=s, i=i)) for k, m in MODELS.items() for s in (1, 2)}
    for (k, s), v in vids.items():
        st = stats(v)
        per_model[k].append(st)
        rows.append({"prompt": i, "model": k, "seed": s, **st})
    for k in MODELS:  # seed diversity: mean abs pixel diff between the two seeds of the same prompt
        d = float(np.abs(vids[(k, 1)].astype(np.float32) - vids[(k, 2)].astype(np.float32)).mean() / 255)
        rows.append({"prompt": i, "model": k, "seed": "1vs2", "seed_pixel_diff": d})
    # contact sheet: rows dmd2 s1, dmd2 s2, ours s1, ours s2; cols frames 0/20/40/60/80
    fr = [0, 20, 40, 60, 80]
    sheet = Image.new("RGB", (60 + W * len(fr), H * 4), "white")
    dr = ImageDraw.Draw(sheet)
    for r, (k, s) in enumerate([("dmd2", 1), ("dmd2", 2), ("ours", 1), ("ours", 2)]):
        dr.text((4, r * H + H // 2 - 6), f"{k}\ns{s}", fill="black")
        for c, f in enumerate(fr):
            sheet.paste(Image.fromarray(vids[(k, s)][f]), (60 + c * W, r * H))
    sheet.save(f"runs/phase0/frames/sheet_prompt{i}.png")

with open("runs/phase0/sample_stats.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
keys = list(per_model["dmd2"][0])
print(f"{'metric':18s} {'dmd2':>8s} {'ours':>8s}   (mean over 4 prompts x 2 seeds)")
for key in keys:
    print(f"{key:18s} {np.mean([s[key] for s in per_model['dmd2']]):8.4f} {np.mean([s[key] for s in per_model['ours']]):8.4f}")
for k in MODELS:
    print(f"seed_pixel_diff {k}: ", [round(r["seed_pixel_diff"], 4) for r in rows if r.get("seed") == "1vs2" and r["model"] == k])
