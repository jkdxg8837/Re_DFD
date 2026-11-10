"""B-pre step 2.3: one technical normalisation for every clip of ViPE / Mixkit / OpenVid.

Per sample of a WDS folder (mp4 + txt): decode all frames (PyAV, rgb24, as `decode_video_full`), take frames [0, 81)
(as `transform_video`), resize the small side + centre-crop to 832x480 with fastgen's own
`resize_small_side_aspect_preserving` / `center_crop` (bicubic, antialias; identical to the training loader), then
re-encode with one fixed setting:
  $FFMPEG_BIN -f rawvideo -pix_fmt rgb24 -s 832x480 -r 16 -i - -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -an
fps is not resampled here: every input clip is already 16 fps (inventory, job 66918); this is asserted.

Writes <out>/<dataset>/<clip_id>.mp4 and <out>/<dataset>_index.jsonl (clip_id, dataset, src_shard, src_key,
caption_orig, n_frames_in, status). Run with the training venv: PYTHONPATH=$PWD .venv/bin/python ...
"""
import argparse
import glob
import io
import json
import os
import subprocess
import tarfile
from multiprocessing import Pool

import av
import numpy as np
import torch

from fastgen.datasets.crop_resize import center_crop, resize_small_side_aspect_preserving

W, H, T, FPS = 832, 480, 81, 16
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")  # system ffmpeg has no libx264; sbatch sets the imageio-ffmpeg 7.0.2 static binary
FFMPEG = [FFMPEG_BIN, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
          "-an", "-threads", "2"]


def decode(data):
    with av.open(io.BytesIO(data)) as c:
        v = c.streams.video[0]
        fps = float(v.average_rate)
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(v)]
    return np.stack(frames), fps


def normalise(frames):
    x = torch.from_numpy(frames[:T]).permute(0, 3, 1, 2)  # t c h w, uint8
    x = resize_small_side_aspect_preserving(x, (W, H))
    x = center_crop(x, (W, H))
    return x.permute(0, 2, 3, 1).contiguous().numpy()  # t h w c, uint8


def do_shard(job):
    path, dataset, out_dir, id_map = job
    rows = []
    groups = {}
    with tarfile.open(path) as tf:
        for m in tf:
            if m.isfile() and "." in os.path.basename(m.name):
                base, ext = m.name.rsplit(".", 1)
                groups.setdefault(base, {})[ext] = tf.extractfile(m).read()
    for key in sorted(groups):
        d = groups[key]
        cid = f"{dataset}_{id_map.get(key, key) if id_map else key}"
        rows.append(process(cid, dataset, d["mp4"], d["txt"].decode("utf-8").strip(), out_dir,
                            src_shard=os.path.basename(path), src_key=key))
    return rows


def process(cid, dataset, mp4, caption, out_dir, **src):
    r = dict(clip_id=cid, dataset=dataset, **src, caption_orig=caption)
    dst = os.path.join(out_dir, dataset, cid + ".mp4")
    try:
        frames, fps = decode(mp4)
        r["n_frames_in"] = int(len(frames))
        assert abs(fps - FPS) < 1e-3, f"fps {fps}"
        if len(frames) < T:
            r["status"] = "drop_short"
        else:
            if not os.path.exists(dst):
                y = normalise(frames)
                tmp = dst + ".tmp.mp4"
                subprocess.run(FFMPEG + [tmp], input=y.tobytes(), check=True)
                os.replace(tmp, dst)
            r["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        r["status"] = "error"
        r["error"] = repr(e)
    return r


def do_prep(job):
    """Prep-directory mode (OpenVid pool): items = (source id, clip path, caption, meta)."""
    items, dataset, out_dir = job
    rows = []
    for sid, path, cap, meta in items:
        with open(path, "rb") as f:
            r = process(f"{dataset}_{sid}", dataset, f.read(), cap, out_dir, src_shard="prep", src_key=sid)
        r["src_duration_s"] = meta["n_src"] / meta["src_fps"]
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--wds", default="")
    ap.add_argument("--prep_dir", default="", help="prep dir with meta.jsonl + clips/ (OpenVid pool) instead of --wds")
    ap.add_argument("--captions_csv", default="data/openvid_prep/OpenVidHD.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--id_map", default="", help="optional json {wds_key: source_id}")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_shards", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, a.dataset), exist_ok=True)
    id_map = json.load(open(a.id_map)) if a.id_map else None
    if a.prep_dir:
        import csv

        csv.field_size_limit(10**9)
        meta = [json.loads(l) for l in open(os.path.join(a.prep_dir, "meta.jsonl"))]
        meta = [m for m in meta if m["status"] == "ok"]
        want = {m["id"] for m in meta}
        caps = {r["video"][:-4]: r["caption"].strip() for r in csv.DictReader(open(a.captions_csv))
                if r["video"][:-4] in want}
        items = [(m["id"], os.path.join(a.prep_dir, "clips", m["id"] + ".mp4"), caps[m["id"]], m) for m in meta]
        jobs = [(items[i: i + 50], a.dataset, a.out) for i in range(0, len(items), 50)]
        fn = do_prep
    else:
        shards = sorted(glob.glob(os.path.join(a.wds, "*.tar")))
        jobs = [(s, a.dataset, a.out, id_map) for s in shards]
        fn = do_shard
    if a.max_shards:
        jobs = jobs[: a.max_shards]
    rows = []
    with Pool(a.workers) as p:
        for i, rs in enumerate(p.imap_unordered(fn, jobs)):
            rows.extend(rs)
            if i % 20 == 0:
                print(f"{a.dataset}: {i + 1}/{len(jobs)} chunks, {len(rows)} samples", flush=True)
    rows.sort(key=lambda r: (r["src_shard"], r["src_key"]))
    assert len({r["clip_id"] for r in rows}) == len(rows), "duplicate clip ids"
    with open(os.path.join(a.out, f"{a.dataset}_index.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(a.dataset, Counter(r["status"] for r in rows))


if __name__ == "__main__":
    main()
