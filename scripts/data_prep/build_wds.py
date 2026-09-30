# SPDX-License-Identifier: Apache-2.0
"""Pack prepared Mixkit clips + Qwen captions into WebDataset shards shaped like ViPE `long` (spec §2.3, §3 E2).

Writes two datasets from the same clips, captions and (seeded) sample order:
  <out>/mixkit_wds/            <key>.mp4 (Mixkit clip), <key>.txt (Qwen caption)                     -> E2
  <out>/mixkit_vipe_pair_wds/  same + <key>.student.mp4 (ViPE `long` video whose caption is the nearest
                               neighbor of the Mixkit caption) + <key>.pair.json                     -> E2s
Nearest neighbor: cosine of sentence-transformers all-mpnet-base-v2 embeddings, over all ViPE `long` captions,
with replacement (reuse statistics are reported). 100 samples per shard, as in ViPE `long`.

Usage: python scripts/data_prep/build_wds.py --captions <mixkit captions.jsonl> [--meta data/mixkit_prep/meta.jsonl]
       [--vipe data/vipe_fliter_complete/long] [--out data] [--seed 0]
Run with .venv-vlm on a GPU node (embedding ~29k captions).
"""

import argparse
import glob
import io
import json
import os
import random
import tarfile
from collections import Counter

import numpy as np


def add(tf, name, data: bytes):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = 0
    tf.addfile(ti, io.BytesIO(data))


def write_shards(samples, out_dir, per_shard=100):
    os.makedirs(out_dir, exist_ok=True)
    assert not glob.glob(os.path.join(out_dir, "*.tar")), f"{out_dir} already has shards; not overwriting"
    for s0 in range(0, len(samples), per_shard):
        with tarfile.open(os.path.join(out_dir, f"{s0 // per_shard:06d}.tar"), "w") as tf:
            for key, files in samples[s0 : s0 + per_shard]:
                for ext, data in files:
                    add(tf, f"{key}.{ext}", data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--meta", default="data/mixkit_prep/meta.jsonl")
    ap.add_argument("--vipe", default="data/vipe_fliter_complete/long")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="", help="appended to both output dir names (existing dirs are never overwritten)")
    a = ap.parse_args()

    ok = {r["id"]: r for r in map(json.loads, open(a.meta)) if r["status"] == "ok"}
    caps = {r["id"]: r["caption"] for r in map(json.loads, open(a.captions)) if r["id"] in ok}
    ids = sorted(caps)
    random.Random(a.seed).shuffle(ids)
    print(f"{len(ok)} ok clips, {len(ids)} with captions", flush=True)

    # ViPE captions and where each video lives
    vipe_keys, vipe_caps, vipe_tar = [], [], []
    for t in sorted(glob.glob(os.path.join(a.vipe, "*.tar"))):
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                if m.name.endswith(".txt"):
                    vipe_keys.append(m.name[:-4])
                    vipe_caps.append(" ".join(tf.extractfile(m).read().decode().split()))
                    vipe_tar.append(t)
    print(f"{len(vipe_keys)} ViPE captions", flush=True)

    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    ev = m.encode(vipe_caps, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    em = m.encode([caps[i] for i in ids], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    sim = em @ ev.T
    nn = sim.argmax(1)
    nn_cos = sim[np.arange(len(ids)), nn]
    reuse = Counter(nn.tolist())
    rand_cos = sim[np.arange(len(ids)), np.random.default_rng(a.seed).integers(0, len(vipe_keys), len(ids))]
    pstats = {
        "n_mixkit": len(ids), "n_vipe": len(vipe_keys), "nn_cos_mean": float(nn_cos.mean()),
        "nn_cos_p5": float(np.percentile(nn_cos, 5)), "nn_cos_p95": float(np.percentile(nn_cos, 95)),
        "random_pair_cos_mean": float(rand_cos.mean()), "n_distinct_vipe": len(reuse),
        "max_reuse": max(reuse.values()), "frac_vipe_used_more_than_once": float(np.mean([c > 1 for c in reuse.values()])),
    }
    print(json.dumps(pstats, indent=1), flush=True)

    vipe_mp4 = {}
    need = {}
    for j in set(nn.tolist()):
        need.setdefault(vipe_tar[j], []).append(j)
    for t, js in need.items():
        with tarfile.open(t) as tf:
            for j in js:
                vipe_mp4[j] = tf.extractfile(vipe_keys[j] + ".mp4").read()

    plain, paired = [], []
    for k, (i, j, c) in enumerate(zip(ids, nn.tolist(), nn_cos.tolist())):
        key = f"{k:08d}"
        mp4 = open(ok[i]["out"], "rb").read()
        txt = caps[i].encode()
        plain.append((key, [("mp4", mp4), ("txt", txt)]))
        pj = json.dumps({"mixkit_id": i, "vipe_key": vipe_keys[j], "vipe_tar": os.path.basename(vipe_tar[j]),
                         "caption_cos": c, "vipe_caption": vipe_caps[j]}).encode()
        paired.append((key, [("mp4", mp4), ("txt", txt), ("student.mp4", vipe_mp4[j]), ("pair.json", pj)]))
    write_shards(plain, os.path.join(a.out, "mixkit_wds" + a.suffix))
    write_shards(paired, os.path.join(a.out, "mixkit_vipe_pair_wds" + a.suffix))
    json.dump({"pairing": pstats, "captions": os.path.abspath(a.captions), "seed": a.seed},
              open(os.path.join(a.out, "mixkit_vipe_pair_wds" + a.suffix, "build_info.json"), "w"), indent=1)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
