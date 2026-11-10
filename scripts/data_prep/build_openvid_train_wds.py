# SPDX-License-Identifier: Apache-2.0
"""Pack the OpenVid training clips (prep_openvid_train.py) into WebDataset shards shaped like data/mixkit_wds_v3
(task2.md B1): <key>.mp4 (clip) + <key>.txt (OpenVid's own caption from OpenVidHD.csv, stripped), keys %08d in a
seeded shuffle (seed 0, as build_wds.py), 100 samples per shard. Never overwrites existing shards.
Also writes <out>/build_info.json (counts, caption word statistics) and <out>/ids.txt (key -> clip id).

Usage: python scripts/data_prep/build_openvid_train_wds.py --prep data/openvid_train_prep --n 5227 --out data/openvid_train_wds
"""

import argparse
import csv
import glob
import io
import json
import os
import random
import tarfile

import numpy as np


def add(tf, name, data: bytes):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = 0
    tf.addfile(ti, io.BytesIO(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default="data/openvid_prep/OpenVidHD.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ids = [l.strip() for l in open(os.path.join(a.prep, f"subset{a.n}.txt")) if l.strip()]
    assert len(ids) == a.n, (len(ids), a.n)
    want = set(ids)
    caps = {}
    with open(a.csv) as f:
        for r in csv.DictReader(f):
            v = r["video"][:-4] if r["video"].endswith(".mp4") else r["video"]
            if v in want:
                caps[v] = r["caption"].strip()
    missing = [i for i in ids if not caps.get(i)]
    assert not missing, f"{len(missing)} clips without caption, e.g. {missing[:3]}"
    random.Random(a.seed).shuffle(ids)
    os.makedirs(a.out, exist_ok=True)
    assert not glob.glob(os.path.join(a.out, "*.tar")), f"{a.out} already has shards; not overwriting"
    for s0 in range(0, len(ids), 100):
        with tarfile.open(os.path.join(a.out, f"{s0 // 100:06d}.tar"), "w") as tf:
            for k in range(s0, min(s0 + 100, len(ids))):
                i = ids[k]
                add(tf, f"{k:08d}.mp4", open(os.path.join(a.prep, "clips", f"{i}.mp4"), "rb").read())
                add(tf, f"{k:08d}.txt", caps[i].encode())
    words = np.array([len(caps[i].split()) for i in ids])
    info = {"n": len(ids), "seed": a.seed, "captions": a.csv, "n_shards": (len(ids) + 99) // 100,
            "caption_words": {"mean": float(words.mean()), "median": float(np.median(words)),
                              "p5": float(np.percentile(words, 5)), "p95": float(np.percentile(words, 95))}}
    json.dump(info, open(os.path.join(a.out, "build_info.json"), "w"), indent=1)
    open(os.path.join(a.out, "ids.txt"), "w").write("\n".join(f"{k:08d} {i}" for k, i in enumerate(ids)) + "\n")
    print(json.dumps(info), flush=True)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
