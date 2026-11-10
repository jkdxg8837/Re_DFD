# SPDX-License-Identifier: Apache-2.0
"""Pick the ViPE `long` subset for teacher-ODE inversion (user, 2026-10-06) and unpack it for scripts/invert_real.py.

Same protocol as the Mixkit E4 subset: a fixed random 1,280-clip subset (seed 0) of the whole ViPE `long` set; clips
must have >= 81 frames (training uses decoder_type="full" + frames 0:81, as invert_real.py). The original ViPE
captions are kept (no recaptioning).

Output:
  <out>/clips/<id>.mp4        clip, byte-identical copy of the WDS member
  <out>/captions.jsonl        {"id", "caption", "shard", "key"} per clip
  <out>/subset1280.txt        clip ids in selection order (the --list of invert_real.py / build_inv_wds.py)
Clip id = vipe-<shard>-<key>, e.g. vipe-000123-00012345. Never overwrites.

Usage: python scripts/data_prep/prep_vipe_inv.py --src data/vipe_fliter_complete/long --out data/vipe_inv_prep
       (no torch; login node is fine)
"""

import argparse
import glob
import json
import os
import random
import tarfile

import av


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1280)
    ap.add_argument("--min_frames", type=int, default=81)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    list_path = os.path.join(a.out, f"subset{a.n}.txt")
    assert not os.path.exists(list_path), f"{list_path} exists; not overwriting"
    os.makedirs(os.path.join(a.out, "clips"), exist_ok=True)

    # index every (shard, key) that has both .mp4 and .txt
    tars = sorted(glob.glob(os.path.join(a.src, "*.tar")))
    pool = []
    for tp in tars:
        with tarfile.open(tp) as tf:
            names = set(tf.getnames())
        shard = os.path.basename(tp)[:-4]
        pool += [(shard, n[:-4]) for n in sorted(names) if n.endswith(".mp4") and n[:-4] + ".txt" in names]
    print(f"{len(pool)} samples in {len(tars)} shards")
    random.Random(a.seed).shuffle(pool)

    picked, short = [], 0
    with open(os.path.join(a.out, "captions.jsonl"), "w") as fc:
        for shard, key in pool:
            if len(picked) == a.n:
                break
            cid = f"vipe-{shard}-{key}"
            with tarfile.open(os.path.join(a.src, f"{shard}.tar")) as tf:
                mp4 = tf.extractfile(f"{key}.mp4").read()
                cap = tf.extractfile(f"{key}.txt").read().decode().strip()
            clip = os.path.join(a.out, "clips", f"{cid}.mp4")
            with open(clip, "wb") as f:
                f.write(mp4)
            with av.open(clip) as c:
                nf = sum(1 for _ in c.decode(video=0))
            if nf < a.min_frames:
                os.remove(clip)
                short += 1
                continue
            fc.write(json.dumps({"id": cid, "caption": cap, "shard": shard, "key": key}) + "\n")
            picked.append(cid)
    assert len(picked) == a.n, f"only {len(picked)} clips with >= {a.min_frames} frames"
    with open(list_path, "w") as f:
        f.write("\n".join(picked) + "\n")
    print(f"picked {len(picked)} clips ({short} dropped as < {a.min_frames} frames) -> {a.out}")


if __name__ == "__main__":
    main()
