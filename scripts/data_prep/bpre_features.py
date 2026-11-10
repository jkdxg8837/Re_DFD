"""B-pre steps 2.1 / 2.4 / 2.5 on the normalised clips (832x480, 16 fps, 81 frames = the training window).

- 2.1 scene cuts: PySceneDetect `detect(path, ContentDetector(threshold=27, min_scene_len=15))` (same detector settings
  as the Mixkit / OpenVid prep); n_cuts = len(scenes) - 1 (0 if no cut found).
- 2.4 motion: OpenCV Farneback (pyr_scale 0.5, levels 3, winsize 15, iterations 3, poly_n 5, poly_sigma 1.2) on
  grayscale frames resized (INTER_AREA) to 128 px short side, pairs (i, i+1) for i = 0, 4, ..., 76 (20 pairs);
  motion = mean flow magnitude over pixels and pairs (pixels at 128-px scale per 1/16 s).
- 2.5 CLIP (diagnostic only): openai/clip-vit-base-patch32, frames 0 / 40 / 80, standard CLIPProcessor
  preprocessing, image embeddings mean-pooled over the 3 frames, l2-normalised -> 512-d float32.

Usage: .venv-bpre/bin/python scripts/data_prep/bpre_features.py --index <clips>/<d>_index.jsonl --clips <clips> \
          --out runs/phaseAB/bpre/features [--workers 16]
Writes <out>/<d>_stats.jsonl (clip_id, n_cuts, motion) and <out>/<d>_clip.npy + <d>_clip_ids.txt.
"""
import argparse
import json
import os
from multiprocessing import Pool

import av
import cv2
import numpy as np

T = 81
FLOW_PAIRS = list(range(0, T - 1, 4))
CLIP_FRAMES = [0, 40, 80]


def read_frames(path):
    with av.open(path) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])])


def motion(frames):
    h, w = frames.shape[1:3]
    s = 128 / min(h, w)
    size = (int(round(w * s)), int(round(h * s)))
    mags = []
    for i in FLOW_PAIRS:
        a = cv2.cvtColor(cv2.resize(frames[i], size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(cv2.resize(frames[i + 1], size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(np.linalg.norm(flow, axis=-1).mean())
    return float(np.mean(mags))


def cpu_job(job):
    cid, path = job
    from scenedetect import ContentDetector, detect

    cv2.setNumThreads(1)
    r = dict(clip_id=cid)
    try:
        scenes = detect(path, ContentDetector(threshold=27.0, min_scene_len=15))
        r["n_cuts"] = max(len(scenes) - 1, 0)
        fr = read_frames(path)
        r["n_frames"] = int(len(fr))
        r["motion"] = motion(fr)
        r["clip_frames"] = fr[CLIP_FRAMES]
    except Exception as e:  # noqa: BLE001
        r["error"] = repr(e)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    a = ap.parse_args()
    import torch
    from transformers import CLIPModel, CLIPProcessor

    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(l) for l in open(a.index)]
    rows = [r for r in rows if r["status"] == "ok"]
    d = rows[0]["dataset"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(a.clip_model).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(a.clip_model)
    jobs = [(r["clip_id"], os.path.join(a.clips, d, r["clip_id"] + ".mp4")) for r in rows]
    ids, feats, buf = [], [], []

    def flush():
        if not buf:
            return
        imgs = [f for _, fr in buf for f in fr]
        with torch.no_grad():
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
            e = model.get_image_features(pixel_values=px)
            e = e if torch.is_tensor(e) else e.pooler_output
            e = e.float().view(len(buf), len(CLIP_FRAMES), -1).mean(1)
            e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
        ids.extend(c for c, _ in buf)
        feats.append(e)
        buf.clear()

    BLOCK = 512  # bounded blocks: imap would otherwise queue all decoded frames while CLIP (CPU) lags (OOM, job 66962)
    with open(os.path.join(a.out, f"{d}_stats.jsonl"), "w") as f, Pool(a.workers) as p:
        for b0 in range(0, len(jobs), BLOCK):
            for i, r in enumerate(p.imap(cpu_job, jobs[b0: b0 + BLOCK], chunksize=4), start=b0):
                fr = r.pop("clip_frames", None)
                f.write(json.dumps(r) + "\n")
                if fr is not None:
                    buf.append((r["clip_id"], list(fr)))
                if len(buf) >= 64:
                    flush()
                if i % 1000 == 0:
                    print(f"{d}: {i + 1}/{len(jobs)}", flush=True)
    flush()
    np.save(os.path.join(a.out, f"{d}_clip.npy"), np.concatenate(feats).astype(np.float32))
    with open(os.path.join(a.out, f"{d}_clip_ids.txt"), "w") as f:
        f.write("\n".join(ids) + "\n")
    print(d, "done", len(ids), "clip features")


if __name__ == "__main__":
    main()
