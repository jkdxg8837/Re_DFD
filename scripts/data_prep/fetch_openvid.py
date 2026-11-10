# SPDX-License-Identifier: Apache-2.0
"""Fetch a random sample of OpenVidHD-0.4M videos without downloading the ~4.9 TB of zips (user, 2026-10-09).

HF `nkp37/OpenVid-1M`, folder OpenVidHD: 14 whole zips + split zips (`*_part_aa`, `*_part_ab`, ... = one zip cut in
pieces). Each logical zip is read through HTTP range requests (a virtual file over its pieces), so only the central
directories and the chosen members are transferred.

  1. index: name -> (logical zip, member) for every logical zip -> <out>/index.jsonl
  2. candidates: CSV rows (data/train/OpenVidHD.csv) with seconds >= --min_sec that are in the index; seeded random
     sample (seed 0) of --k -> <out>/candidates.txt (order = selection order)
  3. download the candidates' mp4s -> <out>/src/<name>  (resumable)

Usage: python scripts/data_prep/fetch_openvid.py --out data/openvid_prep --k 400   (stdlib only; login node is fine)
"""

import argparse
import csv
import json
import os
import random
import re
import time
import urllib.request
import zipfile

REPO = "https://huggingface.co/datasets/nkp37/OpenVid-1M"


def http_get(url, rng=None, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


class RangeFile:
    """Read-only seekable file over the concatenation of remote pieces [(url, size), ...]."""

    def __init__(self, pieces):
        self.pieces, self.pos = pieces, 0
        self.size = sum(s for _, s in pieces)

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out, start = [], self.pos
        base = 0
        for url, s in self.pieces:
            lo, hi = max(start, base), min(start + n, base + s)
            if lo < hi:
                out.append(http_get(url, (lo - base, hi - base - 1)))
            base += s
        self.pos += n
        return b"".join(out)


def logical_zips():
    tree = json.loads(http_get(f"https://huggingface.co/api/datasets/nkp37/OpenVid-1M/tree/main/OpenVidHD"))
    groups = {}
    for f in tree:
        p = f["path"]
        if p.endswith(".zip"):
            groups[p] = [(p, f["size"])]
        else:
            m = re.match(r"(.+)_part_(a[a-z])$", p)
            if m:
                groups.setdefault(m[1], []).append((p, f["size"]))
    out = {}
    for g, ps in groups.items():
        ps.sort()
        out[g] = [(f"{REPO}/resolve/main/{p}", s) for p, s in ps]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=400)
    ap.add_argument("--min_sec", type=float, default=5.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "src"), exist_ok=True)
    zips = logical_zips()
    idx_path = os.path.join(a.out, "index.jsonl")
    done = {json.loads(l)["zip"] for l in open(idx_path)} if os.path.exists(idx_path) else set()
    with open(idx_path, "a") as fo:
        for g, pieces in sorted(zips.items()):
            if g in done:
                continue
            try:
                zf = zipfile.ZipFile(RangeFile(pieces))
            except Exception as e:
                print(f"skip {g}: {e!r}", flush=True)
                fo.write(json.dumps({"zip": g, "error": repr(e)[:200]}) + "\n")
                continue
            names = [i.filename for i in zf.infolist() if i.filename.endswith(".mp4")]
            fo.write(json.dumps({"zip": g, "names": names}) + "\n")
            fo.flush()
            print(f"{g}: {len(names)} mp4", flush=True)
    where = {}
    for l in open(idx_path):
        r = json.loads(l)
        for n in r.get("names", []):
            where[os.path.basename(n)] = (r["zip"], n)
    rows = []
    with open("data/openvid_prep/OpenVidHD.csv") as f:
        for r in csv.DictReader(f):
            if float(r["seconds"] or 0) >= a.min_sec and r["video"] in where:
                rows.append(r["video"])
    print(f"index: {len(where)} mp4 in {len(zips)} zips; eligible (>= {a.min_sec} s, indexed): {len(rows)}", flush=True)
    cand_path = os.path.join(a.out, "candidates.txt")
    if os.path.exists(cand_path):
        cands = [l.strip() for l in open(cand_path) if l.strip()]
    else:
        cands = random.Random(a.seed).sample(sorted(rows), a.k)
        open(cand_path, "w").write("\n".join(cands) + "\n")
    open_zips = {}
    for i, v in enumerate(cands):
        dst = os.path.join(a.out, "src", v)
        if os.path.exists(dst):
            continue
        g, member = where[v]
        if g not in open_zips:
            open_zips[g] = zipfile.ZipFile(RangeFile(zips[g]))
        data = open_zips[g].read(member)
        with open(dst + ".tmp", "wb") as f:
            f.write(data)
        os.replace(dst + ".tmp", dst)
        if i % 25 == 0:
            print(f"[{i + 1}/{len(cands)}] {v} {len(data) / 1e6:.1f} MB", flush=True)
    print("FETCH_DONE", flush=True)


if __name__ == "__main__":
    main()
