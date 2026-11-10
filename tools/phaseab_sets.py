# SPDX-License-Identifier: Apache-2.0
"""Phase 0.1 / 0.2 (task2.md): reference-set manifests and the 200-prompt set.

Decisions (user 2026-10-21, task2.md "Answers"):
  R_mx   = 300 random clips (seed 0) of the Mixkit training pool data/mixkit_wds_v3 (Qwen v3 captions)
  S_vipe = 300 random clips (seed 0) of ViPE `long` (data/vipe_fliter_complete/long)
  R_ov   = the 256 clips of data/openvid_prep/subset256.txt (earlier OpenVidHD held-out test) + 44 random (seed 0)
           of the other prepared candidates in data/openvid_prep/clips (OpenVid's own captions)
Prompts (0.2): 100 ViPE captions from the ViPE pool minus S_vipe, 50 Mixkit captions from the pool minus R_mx,
50 OpenVid captions from the prepared OpenVid candidates minus R_ov (all seed 0); prompt seed = 1000 + prompt_idx.

Outputs under runs/phaseAB/:
  sets/<set>.jsonl   {"idx", "id", "mp4", "caption"}; WDS clips are extracted (byte copy) to clips/<set>/<id>.mp4
  prompts.jsonl      {"prompt_idx", "source", "src_id", "prompt", "seed"}
Run in a Slurm CPU job (reads ~46 GB of WDS tars): python tools/phaseab_sets.py
"""

from __future__ import annotations

import glob
import json
import os
import random
import tarfile

OUT = "runs/phaseAB"
N_REF = 300


def wds_index(pattern: str, tag: str):
    """All (id, tar, key) of the mp4 samples in the WDS shards; id = <tag>-<shard>-<key> (as data/vipe_inv_prep)."""
    out = []
    for t in sorted(glob.glob(pattern)):
        shard = os.path.basename(t)[:-4]
        with tarfile.open(t) as tf:
            for n in sorted(tf.getnames()):
                if n.endswith(".mp4"):
                    out.append((f"{tag}-{shard}-{n[:-4]}", t, n[:-4]))
    return out


def extract(items, set_name):
    """Byte-copy mp4 + caption of the chosen WDS samples; returns manifest rows."""
    d = os.path.join(OUT, "clips", set_name)
    os.makedirs(d, exist_ok=True)
    by_tar = {}
    for i, (cid, t, key) in enumerate(items):
        by_tar.setdefault(t, []).append((i, cid, key))
    rows = [None] * len(items)
    for t, lst in by_tar.items():
        with tarfile.open(t) as tf:
            for i, cid, key in lst:
                p = os.path.join(d, f"{cid}.mp4")
                with open(p, "wb") as f:
                    f.write(tf.extractfile(f"{key}.mp4").read())
                cap = tf.extractfile(f"{key}.txt").read().decode().strip()
                rows[i] = {"idx": i, "id": cid, "mp4": p, "caption": cap}
    return rows


def captions_of(items):
    by_tar = {}
    for i, (cid, t, key) in enumerate(items):
        by_tar.setdefault(t, []).append((i, cid, key))
    caps = [None] * len(items)
    for t, lst in by_tar.items():
        with tarfile.open(t) as tf:
            for i, cid, key in lst:
                caps[i] = (cid, tf.extractfile(f"{key}.txt").read().decode().strip())
    return caps


def main():
    os.makedirs(os.path.join(OUT, "sets"), exist_ok=True)
    prompts = []

    def write(name, rows):
        with open(os.path.join(OUT, "sets", f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} clips")

    # ViPE long
    vipe = wds_index("data/vipe_fliter_complete/long/*.tar", "vipe")
    rng = random.Random(0)
    perm = rng.sample(range(len(vipe)), len(vipe))
    write("S_vipe", extract([vipe[i] for i in perm[:N_REF]], "S_vipe"))
    vp = captions_of([vipe[i] for i in perm[N_REF:N_REF + 100]])
    print(f"ViPE pool {len(vipe)}")

    # Mixkit pool (E2 training data)
    mx = wds_index("data/mixkit_wds_v3/*.tar", "mixkit")
    rng = random.Random(0)
    perm = rng.sample(range(len(mx)), len(mx))
    write("R_mx", extract([mx[i] for i in perm[:N_REF]], "R_mx"))
    mp = captions_of([mx[i] for i in perm[N_REF:N_REF + 50]])
    print(f"Mixkit pool {len(mx)}")

    # OpenVid: prepared candidates (own captions from OpenVidHD.csv)
    import csv

    clips = sorted(f[:-4] for f in os.listdir("data/openvid_prep/clips") if f.endswith(".mp4"))
    s256 = [l.strip() for l in open("data/openvid_prep/subset256.txt") if l.strip()]
    assert set(s256) <= set(clips)
    rest = [c for c in clips if c not in set(s256)]
    rng = random.Random(0)
    rest = rng.sample(rest, len(rest))
    need = set(clips)
    caps = {}
    with open("data/openvid_prep/OpenVidHD.csv", newline="") as f:
        for r in csv.DictReader(f):
            k = r["video"][:-4] if r["video"].endswith(".mp4") else r["video"]
            if k in need:
                caps[k] = r["caption"].strip()
    assert set(clips) <= set(caps), f"missing OpenVid captions: {sorted(set(clips) - set(caps))[:5]}"
    r_ov = s256 + rest[: N_REF - len(s256)]
    write("R_ov", [{"idx": i, "id": c, "mp4": f"data/openvid_prep/clips/{c}.mp4", "caption": caps[c]}
                   for i, c in enumerate(r_ov)])
    op = [(c, caps[c]) for c in rest[N_REF - len(s256):][:50]]
    print(f"OpenVid prepared {len(clips)}, R_ov {len(r_ov)}, left for prompts {len(rest) - (N_REF - len(s256))}")
    assert len(op) == 50

    for src, lst in [("vipe", vp), ("mixkit", mp), ("openvid", op)]:
        for cid, cap in lst:
            k = len(prompts)
            prompts.append({"prompt_idx": k, "source": src, "src_id": cid, "prompt": cap, "seed": 1000 + k})
    with open(os.path.join(OUT, "prompts.jsonl"), "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    print(f"prompts: {len(prompts)}")


if __name__ == "__main__":
    main()
