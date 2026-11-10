# SPDX-License-Identifier: Apache-2.0
"""OpenVidHD training shard for Phase B (task2.md B1, run B-OV-*): the Mixkit / OpenVid-eval pipeline
(prep_mixkit.py steps 1-6: drop portrait, 1280x720, nearest-frame 16 fps, middle 81 frames, drop PySceneDetect cuts,
libx264 crf 18) applied to a fresh seeded sample of OpenVidHD, disjoint from the whole OpenVidHD held-out preparation.
Reused (as a new file) from the 2026-10-20 E2o pool script, which was validated frame for frame against
prep_mixkit.process; the only change is the default of --exclude_src_of (all 400 candidates, not only the 256 clips).

Candidates (written once to <out>/candidates.txt, selection order):
  - OpenVidHD.csv rows with seconds >= --min_sec that are in the zip index (data/openvid_prep/index.jsonl);
  - minus every video in --exclude_list (the 400 held-out candidates of data/openvid_prep, which contain R_ov and the
    256 OpenVidHD eval clips) and every video whose source video (YouTube id = first 11 characters of the name;
    magictime_<id>_... -> <id>) is the source of any video in --exclude_src_of (default: the same 400);
  - seeded shuffle (--seed), then at most one video per source video (as Mixkit: one clip per source id); first --k.
Processing (resumable, <out>/meta.jsonl): each worker range-reads one source mp4 from its HF zip into node-local temp
(--tmp), cuts the clip into <out>/clips/<id>.mp4 and deletes the temp file, so no source video is stored on /dcs/large.
`process_lowmem` reproduces prep_mixkit.process frame for frame but decodes twice (count, then keep only the 81 chosen
frames) instead of holding every 1280x720 source frame in memory (sources up to 55 s).
Stops once the first --n candidates in selection order with status ok are known; writes <out>/subset<n>.txt.

Usage: .venv-prep/bin/python scripts/data_prep/prep_openvid_train.py --out data/openvid_train_prep --k 6000 --n 5227 \
           --workers 16 --tmp $TMPDIR/openvid_src
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys
import zipfile
from collections import Counter
from fractions import Fraction
from multiprocessing import Pool

import av
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_openvid import RangeFile, logical_zips  # noqa: E402
from prep_mixkit import H, N_FRAMES, OUT_FPS, W, n_scene_cuts  # noqa: E402

PREP = "data/openvid_prep"  # eval-sample prep: CSV, zip index, candidates, subset256 (read only)


def source_of(name: str) -> str:
    s = name[:-4] if name.endswith(".mp4") else name
    return s.split("_")[1] if s.startswith("magictime_") else s[:11]


def process_lowmem(args):
    """Same output as prep_mixkit.process((sid, path, out_dir)); two decode passes, 81 frames in memory."""
    sid, path, out_dir = args
    rec = {"id": sid, "src": os.path.relpath(path), "category": os.path.basename(os.path.dirname(path))}
    try:
        with av.open(path) as c:
            v = c.streams.video[0]
            rec.update(width=v.width, height=v.height, src_fps=float(v.average_rate))
            if v.height > v.width:
                return {**rec, "status": "drop_portrait"}
            n_src = sum(1 for _ in c.decode(v))
        fps = rec["src_fps"]
        n_out = int(np.floor(n_src / fps * OUT_FPS + 1e-6))
        rec.update(n_src=n_src, n_out16=n_out)
        if n_out < N_FRAMES:
            return {**rec, "status": "drop_short"}
        start = (n_out - N_FRAMES) // 2
        idx = [min(n_src - 1, int(round(k * fps / OUT_FPS))) for k in range(start, start + N_FRAMES)]
        need, got = set(idx), {}
        with av.open(path) as c:
            v = c.streams.video[0]
            for i, f in enumerate(c.decode(v)):
                if i in need:
                    got[i] = f.to_ndarray(width=W, height=H, format="rgb24", interpolation="AREA")
                if i >= idx[-1]:
                    break
        clip = [got[i] for i in idx]
        del got
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


_ZIPS, _OPEN = None, {}


def _init(zips):
    global _ZIPS
    _ZIPS = zips


def fetch_and_process(task):
    """Range-read one member into node-local temp, cut the clip, delete the temp file."""
    name, zip_name, member, tmp_dir, out_dir = task
    os.makedirs(tmp_dir, exist_ok=True)
    src = os.path.join(tmp_dir, name)
    try:
        if zip_name not in _OPEN:
            _OPEN[zip_name] = zipfile.ZipFile(RangeFile(_ZIPS[zip_name]))
        data = _OPEN[zip_name].read(member)
        with open(src, "wb") as f:
            f.write(data)
        rec = process_lowmem((name[:-4], src, out_dir))
        rec.update(src=f"{zip_name}:{member}", src_mb=round(len(data) / 1e6, 2))
        rec.pop("category", None)
        return rec
    except Exception as e:
        return {"id": name[:-4], "src": f"{zip_name}:{member}", "status": "drop_fetch_error", "error": repr(e)[:300]}
    finally:
        if os.path.exists(src):
            os.remove(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6000, help="number of candidates (oversampled)")
    ap.add_argument("--n", type=int, default=5227, help="clips to keep (first n ok in selection order)")
    ap.add_argument("--min_sec", type=float, default=5.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exclude_list", default=f"{PREP}/candidates.txt")
    ap.add_argument("--exclude_src_of", default=f"{PREP}/candidates.txt")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tmp", required=True, help="node-local temp dir for source mp4s")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "clips"), exist_ok=True)

    where = {}
    for l in open(f"{PREP}/index.jsonl"):
        r = json.loads(l)
        for m in r.get("names", []):
            where[os.path.basename(m)] = (r["zip"], m)
    cand_path = os.path.join(a.out, "candidates.txt")
    if os.path.exists(cand_path):
        cands = [l.strip() for l in open(cand_path) if l.strip()]
    else:
        rows = []
        with open(f"{PREP}/OpenVidHD.csv") as f:
            for r in csv.DictReader(f):
                if float(r["seconds"] or 0) >= a.min_sec and r["video"] in where:
                    rows.append(r["video"])
        excl = {l.strip() for l in open(a.exclude_list) if l.strip()}
        excl = {v if v.endswith(".mp4") else v + ".mp4" for v in excl}
        excl_src = {source_of(l.strip()) for l in open(a.exclude_src_of) if l.strip()}
        n_list = sum(v in excl for v in rows)
        n_src = sum(v not in excl and source_of(v) in excl_src for v in rows)
        pool = sorted(v for v in rows if v not in excl and source_of(v) not in excl_src)
        random.Random(a.seed).shuffle(pool)
        seen, cands = set(), []
        for v in pool:
            s = source_of(v)
            if s not in seen:
                seen.add(s)
                cands.append(v)
            if len(cands) == a.k:
                break
        info = {"eligible": len(rows), "excluded_in_list": n_list, "excluded_same_source_as_eval": n_src,
                "pool_after_exclusion": len(pool), "k": len(cands), "seed": a.seed, "min_sec": a.min_sec,
                "exclude_list": a.exclude_list, "exclude_src_of": a.exclude_src_of, "n_eval_sources": len(excl_src)}
        print(json.dumps(info), flush=True)
        json.dump(info, open(os.path.join(a.out, "selection.json"), "w"), indent=1)
        open(cand_path, "w").write("\n".join(cands) + "\n")

    meta_path = os.path.join(a.out, "meta.jsonl")
    recs = {}
    if os.path.exists(meta_path):
        for l in open(meta_path):
            r = json.loads(l)
            recs[r["id"]] = r

    def n_ok_prefix():
        """ok count over the leading run of processed candidates (selection order)."""
        n = 0
        for v in cands:
            r = recs.get(v[:-4])
            if r is None:
                break
            n += r["status"] == "ok"
        return n

    if n_ok_prefix() < a.n:
        todo = [(v, *where[v], a.tmp, a.out) for v in cands if v[:-4] not in recs]
        print(f"{len(recs)} done, {len(todo)} to process", flush=True)
        with Pool(a.workers, initializer=_init, initargs=(logical_zips(),)) as pool, open(meta_path, "a") as fo:
            for i, rec in enumerate(pool.imap(fetch_and_process, todo, chunksize=1)):
                recs[rec["id"]] = rec
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                if i % 100 == 0:
                    print(i, rec["id"], rec["status"], rec.get("src_mb"), flush=True)
                if n_ok_prefix() >= a.n:
                    pool.terminate()
                    break
    shutil.rmtree(a.tmp, ignore_errors=True)
    proc = [recs[v[:-4]] for v in cands if v[:-4] in recs]
    print("status counts:", dict(Counter(r["status"] for r in proc)), flush=True)
    ok = [r for r in proc if r["status"] == "ok"][: a.n]
    if len(ok) < a.n:
        sys.exit(f"only {len(ok)} usable clips < {a.n}; raise --k")
    last = cands.index(ok[-1]["id"] + ".mp4")
    used = proc[: last + 1]
    print(f"first {a.n} ok reached at candidate {last + 1}; status counts up to there:",
          dict(Counter(r["status"] for r in used)), flush=True)
    open(os.path.join(a.out, f"subset{a.n}.txt"), "w").write("\n".join(r["id"] for r in ok) + "\n")
    print(f"kept {len(ok)} clips -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
