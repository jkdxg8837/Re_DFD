# SPDX-License-Identifier: Apache-2.0
"""Turn FastVideo/Mixkit-Src clips into 81-frame, 16 fps clips stored like ViPE `long` (spec §2.3).

Per source video (at most one clip per Mixkit source id, spec §2.3 clip rules):
  1. drop portrait videos (height > width);
  2. decode at 1280x720 (ViPE `long` storage size; the training loader resizes the small side to 480 and
     center-crops to 832x480, so both datasets go through the same online resize);
  3. resample to 16 fps by nearest-frame selection (source frame round(k * src_fps / 16) for output frame k);
  4. take the middle 81 frames (the loader with decoder_type=full uses frames [0, 81) of the stored clip);
  5. drop clips with a scene cut inside those 81 frames (PySceneDetect ContentDetector, default threshold);
  6. encode h264 yuv420p (libx264, crf 18) to <out>/clips/<id>.mp4.
Writes one JSON line per source video to <out>/meta.jsonl (kept and dropped, with the reason).

Usage: python scripts/data_prep/prep_mixkit.py --src data/mixkit_src --out data/mixkit_prep --workers 24
Runs with .venv-vlm (PyAV + scenedetect). Resumable: ids already in meta.jsonl are skipped.
"""

import argparse
import glob
import json
import os
import re
from fractions import Fraction
from multiprocessing import Pool

import av
import numpy as np

OUT_FPS = 16
N_FRAMES = 81
W, H = 1280, 720
CLIP_RE = re.compile(r"^(?P<src>.+)_clip_(?P<idx>\d+)\.mp4$")


def list_sources(src_root):
    """One clip per Mixkit source id: the lowest clip index, first category folder in sorted order."""
    best = {}
    for p in sorted(glob.glob(os.path.join(src_root, "*", "*.mp4"))):
        m = CLIP_RE.match(os.path.basename(p))
        sid, idx = (m["src"], int(m["idx"])) if m else (os.path.basename(p)[:-4], 0)
        if sid not in best or idx < best[sid][1]:
            best[sid] = (p, idx)
    return [(sid, p) for sid, (p, _) in sorted(best.items())]


def n_scene_cuts(frames):
    from scenedetect.common import FrameTimecode
    from scenedetect.detectors import ContentDetector

    det = ContentDetector()  # threshold 27, min_scene_len 15 (PySceneDetect defaults)
    cuts = []
    for i, f in enumerate(frames):
        cuts += det.process_frame(FrameTimecode(i, fps=float(OUT_FPS)), np.ascontiguousarray(f[:, :, ::-1]))  # RGB->BGR
    return len([c for c in cuts if c.frame_num > 0])


def process(args):
    sid, path, out_dir = args
    rec = {"id": sid, "src": os.path.relpath(path), "category": os.path.basename(os.path.dirname(path))}
    try:
        with av.open(path) as c:
            v = c.streams.video[0]
            rec.update(width=v.width, height=v.height, src_fps=float(v.average_rate))
            if v.height > v.width:
                return {**rec, "status": "drop_portrait"}
            frames = [f.to_ndarray(width=W, height=H, format="rgb24", interpolation="AREA") for f in c.decode(v)]
        n_src, fps = len(frames), rec["src_fps"]
        n_out = int(np.floor(n_src / fps * OUT_FPS + 1e-6))
        rec.update(n_src=n_src, n_out16=n_out)
        if n_out < N_FRAMES:
            return {**rec, "status": "drop_short"}
        start = (n_out - N_FRAMES) // 2
        idx = [min(n_src - 1, int(round(k * fps / OUT_FPS))) for k in range(start, start + N_FRAMES)]
        clip = [frames[i] for i in idx]
        del frames
        rec.update(start16=start, n_cuts=n_scene_cuts(clip))
        if rec["n_cuts"] > 0:
            return {**rec, "status": "drop_scene_cut"}
        out = os.path.join(out_dir, "clips", f"{sid}.mp4")
        tmp = out + ".tmp.mp4"
        with av.open(tmp, "w") as oc:
            s = oc.add_stream("libx264", rate=Fraction(OUT_FPS, 1), options={"crf": "18", "preset": "medium"})
            s.width, s.height, s.pix_fmt = W, H, "yuv420p"
            for f in clip:
                for pkt in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                    oc.mux(pkt)
            for pkt in s.encode():
                oc.mux(pkt)
        os.replace(tmp, out)
        return {**rec, "status": "ok", "out": os.path.relpath(out)}
    except Exception as e:  # corrupt / undecodable source
        return {**rec, "status": "drop_error", "error": repr(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/mixkit_src")
    ap.add_argument("--out", default="data/mixkit_prep")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "clips"), exist_ok=True)
    meta = os.path.join(a.out, "meta.jsonl")
    done = set()
    if os.path.exists(meta):
        done = {json.loads(l)["id"] for l in open(meta)}
    todo = [(sid, p, a.out) for sid, p in list_sources(a.src) if sid not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(done)} done, {len(todo)} to process", flush=True)
    with Pool(a.workers) as pool, open(meta, "a") as fo:
        for i, rec in enumerate(pool.imap_unordered(process, todo, chunksize=4)):
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            if i % 200 == 0:
                print(i, rec["id"], rec["status"], flush=True)
    recs = [json.loads(l) for l in open(meta)]
    from collections import Counter

    print("status counts:", dict(Counter(r["status"] for r in recs)), flush=True)


if __name__ == "__main__":
    main()
