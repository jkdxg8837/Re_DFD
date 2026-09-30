# SPDX-License-Identifier: Apache-2.0
"""Pack the E4 dataset: Mixkit clips of the inversion subset + v3 captions + teacher-ODE inverted noise.

<out>/<key>.mp4 (clip), <key>.txt (Qwen v3 caption), <key>.z_inv.npy (float32 [16, 21, 60, 104], scripts/invert_real.py),
<key>.json (clip id + inversion stats). Seeded shuffle, 100 samples per shard, as build_wds.py. Never overwrites.

Usage: python scripts/data_prep/build_inv_wds.py --list data/mixkit_inv/subset1280.txt --inv data/mixkit_inv \
         --out data/mixkit_inv_wds_v3   (no torch; login node is fine)
"""

import argparse
import glob
import io
import json
import os
import random
import tarfile


def add(tf, name, data: bytes):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = 0
    tf.addfile(ti, io.BytesIO(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--inv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips_dir", default="data/mixkit_prep/clips")
    ap.add_argument("--captions", default="data/mixkit_prep/captions_v3.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.list) if l.strip()]
    caps = {r["id"]: r["caption"] for r in map(json.loads, open(a.captions))}
    meta = {}
    for f in glob.glob(os.path.join(a.inv, "meta_s*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            meta[r["id"]] = r
    missing = [i for i in ids if i not in meta or not os.path.exists(os.path.join(a.inv, "z", f"{i}.npy"))]
    assert not missing, f"{len(missing)} clips not inverted yet, e.g. {missing[:3]}"
    random.Random(a.seed).shuffle(ids)
    os.makedirs(a.out, exist_ok=True)
    assert not glob.glob(os.path.join(a.out, "*.tar")), f"{a.out} already has shards; not overwriting"
    for s0 in range(0, len(ids), 100):
        with tarfile.open(os.path.join(a.out, f"{s0 // 100:06d}.tar"), "w") as tf:
            for k, cid in enumerate(ids[s0 : s0 + 100], start=s0):
                key = f"{k:08d}"
                add(tf, f"{key}.mp4", open(os.path.join(a.clips_dir, f"{cid}.mp4"), "rb").read())
                add(tf, f"{key}.txt", caps[cid].encode())
                add(tf, f"{key}.z_inv.npy", open(os.path.join(a.inv, "z", f"{cid}.npy"), "rb").read())
                add(tf, f"{key}.json", json.dumps(meta[cid]).encode())
    print(f"wrote {len(ids)} samples to {a.out}")


if __name__ == "__main__":
    main()
