# SPDX-License-Identifier: Apache-2.0
"""OpenVidHD candidates -> 81-frame 16 fps clips, exactly the Mixkit pipeline (scripts/data_prep/prep_mixkit.py
`process`: drop portrait, 1280x720, nearest-frame 16 fps resample, middle 81 frames, drop PySceneDetect cuts,
libx264 crf 18), user 2026-10-09. Captions: the dataset's own (OpenVidHD.csv), no recaptioning.

Writes <out>/clips/<id>.mp4, <out>/meta.jsonl (all candidates with status), and for the first --n clips with status
ok in candidate order: <out>/subset<n>.txt and <out>/captions.jsonl ({"id", "caption"}). id = video name without .mp4.
Usage: .venv-prep/bin/python scripts/data_prep/prep_openvid.py --out data/openvid_prep --n 256 --workers 16
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_mixkit import process  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "clips"), exist_ok=True)
    cands = [l.strip() for l in open(os.path.join(a.out, "candidates.txt")) if l.strip()]
    todo = [(v[:-4], os.path.join(a.out, "src", v), a.out) for v in cands]
    with Pool(a.workers) as pool:
        recs = pool.map(process, todo, chunksize=2)
    with open(os.path.join(a.out, "meta.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print("status counts:", dict(Counter(r["status"] for r in recs)))
    ok = [r["id"] for r in recs if r["status"] == "ok"][: a.n]
    assert len(ok) == a.n, f"only {len(ok)} usable clips"
    caps = {}
    with open(os.path.join(a.out, "OpenVidHD.csv")) as f:
        want = {i + ".mp4" for i in ok}
        for r in csv.DictReader(f):
            if r["video"] in want:
                caps[r["video"][:-4]] = r["caption"]
    with open(os.path.join(a.out, f"subset{a.n}.txt"), "w") as f:
        f.write("\n".join(ok) + "\n")
    with open(os.path.join(a.out, "captions.jsonl"), "w") as f:
        for i in ok:
            f.write(json.dumps({"id": i, "caption": caps[i]}) + "\n")
    print(f"kept {len(ok)} clips -> {a.out}")


if __name__ == "__main__":
    main()
