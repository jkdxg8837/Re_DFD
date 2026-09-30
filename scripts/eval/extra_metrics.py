# SPDX-License-Identifier: Apache-2.0
"""Saturation and seed-diversity metrics on the VBench subset videos (spec §6 extra metrics, 2026-09-25).

Input: `runs/eval/<tag>/videos/<prompt>-<k>.mp4` (5 seeds per prompt, identical noise across models).
Per video (all frames, downsampled to 208x120, same as runs/phase0/sample_stats.py):
    sat_mean       mean HSV saturation
    frac_sat_gt_08 fraction of pixels with saturation > 0.8
    clip_frac      fraction of pixel values < 0.02 or > 0.98
Per prompt (the 5 seed videos):
    div_dino       mean pairwise cosine distance (1 - cos) of video embeddings, where a video embedding is the mean
                   of L2-normalised DINOv2 ViT-B/14 CLS features of 9 evenly spaced frames (224x224 center crop of
                   the short-side-resized frame)

Output: runs/eval/<tag>/extra/extra_metrics.json = {"per_video": {name: {...}}, "per_prompt": {prompt: {...}},
         "summary": {...}}. Run in `.venv` on a GPU node.
"""

import argparse
import json
import os
import re
from itertools import combinations

import av
import numpy as np
import torch
import torch.nn.functional as F

DINO = "vit_base_patch14_dinov2.lvd142m"


def read_frames(path: str) -> np.ndarray:
    with av.open(path) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])  # T,H,W,3 uint8


def saturation_stats(frames: np.ndarray) -> dict:
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(120, 208), mode="area")
    mx, mn = x.max(1).values, x.min(1).values
    sat = torch.where(mx > 0, (mx - mn) / mx.clamp_min(1e-6), torch.zeros_like(mx))
    return {"sat_mean": sat.mean().item(), "frac_sat_gt_08": (sat > 0.8).float().mean().item(),
            "clip_frac": ((x < 0.02) | (x > 0.98)).float().mean().item()}


@torch.no_grad()
def dino_embed(model, frames: np.ndarray, device) -> torch.Tensor:
    idx = np.linspace(0, len(frames) - 1, 9).round().astype(int)
    x = torch.from_numpy(frames[idx]).permute(0, 3, 1, 2).float().to(device) / 255.0
    h, w = x.shape[-2:]
    s = 224 / min(h, w)
    x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bicubic", align_corners=False, antialias=True)
    h, w = x.shape[-2:]
    x = x[..., (h - 224) // 2:(h - 224) // 2 + 224, (w - 224) // 2:(w - 224) // 2 + 224]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = model.forward_features((x - mean) / std)[:, 0]  # CLS token
    return F.normalize(F.normalize(feats, dim=-1).mean(0), dim=0).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--eval_root", default="runs/eval")
    a = ap.parse_args()
    import timm

    device = torch.device("cuda")
    model = timm.create_model(DINO, pretrained=True, num_classes=0, img_size=224).eval().to(device)
    for tag in a.tags:
        vdir = os.path.join(a.eval_root, tag, "videos")
        names = sorted(f for f in os.listdir(vdir) if f.endswith(".mp4"))
        per_video, emb = {}, {}
        for n in names:
            prompt, k = re.match(r"(.*)-(\d+)\.mp4$", n).groups()
            fr = read_frames(os.path.join(vdir, n))
            per_video[n] = {"prompt": prompt, "seed": int(k), "n_frames": int(len(fr)), **saturation_stats(fr)}
            emb.setdefault(prompt, {})[int(k)] = dino_embed(model, fr, device)
        per_prompt = {}
        for p, e in emb.items():
            vs = [e[k] for k in sorted(e)]
            d = [1 - float(torch.dot(u, v)) for u, v in combinations(vs, 2)]
            per_prompt[p] = {"n_seeds": len(vs), "div_dino": float(np.mean(d))}
        summary = {m: float(np.mean([v[m] for v in per_video.values()])) for m in ("sat_mean", "frac_sat_gt_08", "clip_frac")}
        summary["div_dino"] = float(np.mean([v["div_dino"] for v in per_prompt.values()]))
        summary["n_videos"], summary["n_prompts"] = len(per_video), len(per_prompt)
        out = os.path.join(a.eval_root, tag, "extra")
        os.makedirs(out, exist_ok=True)
        json.dump({"feature_model": DINO, "per_video": per_video, "per_prompt": per_prompt, "summary": summary},
                  open(os.path.join(out, "extra_metrics.json"), "w"), indent=1)
        print(tag, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
