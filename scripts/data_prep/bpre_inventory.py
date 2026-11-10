"""B-pre step 1 inventory: per-clip container/stream metadata and caption length for every sample of a WDS folder.

Reads tar members in place (no extraction); opens each mp4 with PyAV to read stream metadata only (no full decode).
Output: one jsonl row per sample + a summary json.

Usage: python scripts/data_prep/bpre_inventory.py --name vipe --wds data/vipe_fliter_complete/long --out runs/phaseAB/bpre/inventory
"""
import argparse
import glob
import io
import json
import os
import re
import tarfile
from collections import Counter
from multiprocessing import Pool

import av
import numpy as np


def probe(data):
    with av.open(io.BytesIO(data)) as c:
        v = c.streams.video[0]
        rate = float(v.average_rate) if v.average_rate else None
        dur = float(v.duration * v.time_base) if v.duration else (c.duration / 1e6 if c.duration else None)
        n = v.frames or (round(dur * rate) if dur and rate else None)
        return dict(width=v.codec_context.width, height=v.codec_context.height, fps=rate, n_frames=n, duration=dur,
                    codec=v.codec_context.name, pix_fmt=v.codec_context.pix_fmt, container=c.format.name,
                    n_audio=len(c.streams.audio), bitrate_kbps=len(data) * 8 / 1000 / dur if dur else None)


def do_shard(path):
    rows, groups = [], {}
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            base, ext = m.name.rsplit(".", 1) if "." in os.path.basename(m.name) else (m.name, "")
            groups.setdefault(base, {})[ext] = tf.extractfile(m).read()
    for key, d in groups.items():
        r = dict(shard=os.path.basename(path), key=key, exts=sorted(d))
        if "txt" in d:
            t = d["txt"].decode("utf-8").strip()
            r["words"] = len(t.split())
            r["sentences"] = len([s for s in re.split(r"[.!?]+\s", t) if s.strip()])
        if "mp4" in d:
            r["mp4_bytes"] = len(d["mp4"])
            try:
                r.update(probe(d["mp4"]))
            except Exception as e:  # noqa: BLE001
                r["error"] = repr(e)
        for ext, b in d.items():
            if ext not in ("mp4", "txt"):
                r[f"bytes_{ext}"] = len(b)
        rows.append(r)
    return path, len(groups), rows


def q(x):
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if len(x) == 0:
        return None
    return dict(n=int(len(x)), min=float(x.min()), p5=float(np.percentile(x, 5)), median=float(np.median(x)),
                mean=float(x.mean()), p95=float(np.percentile(x, 95)), max=float(x.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--wds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(a.wds, "*.tar")))
    rows, sizes = [], {}
    with Pool(a.workers) as p:
        for path, n, rs in p.imap_unordered(do_shard, shards):
            sizes[os.path.basename(path)] = n
            rows.extend(rs)
    rows.sort(key=lambda r: (r["shard"], r["key"]))
    with open(os.path.join(a.out, f"{a.name}_clips.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = dict(
        name=a.name, wds=a.wds, n_shards=len(shards), n_samples=len(rows),
        shard_size=q(list(sizes.values())),
        ext_sets=Counter(",".join(r["exts"]) for r in rows).most_common(),
        key_examples=[r["key"] for r in rows[:3]],
        resolution=Counter(f"{r.get('width')}x{r.get('height')}" for r in rows).most_common(10),
        fps=Counter(round(r["fps"], 3) if r.get("fps") else None for r in rows).most_common(10),
        codec=Counter(f"{r.get('container')}/{r.get('codec')}/{r.get('pix_fmt')}" for r in rows).most_common(5),
        audio=Counter(r.get("n_audio") for r in rows).most_common(),
        n_frames=q([r.get("n_frames") for r in rows]),
        duration_s=q([r.get("duration") for r in rows]),
        bitrate_kbps=q([r.get("bitrate_kbps") for r in rows]),
        words=q([r.get("words") for r in rows]),
        sentences=q([r.get("sentences") for r in rows]),
        n_below_81=sum(1 for r in rows if (r.get("n_frames") or 0) < 81),
        errors=sum(1 for r in rows if "error" in r),
    )
    with open(os.path.join(a.out, f"{a.name}_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
