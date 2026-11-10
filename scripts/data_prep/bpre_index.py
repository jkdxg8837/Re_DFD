"""B-pre step 3.6: one per-clip table -> runs/phaseAB/bpre/index.parquet.

Columns: clip_id, dataset, caption_orig, caption_uni, word_count_uni, word_count_orig, motion, n_cuts, n_frames,
src_duration_s (clip duration before trimming), fix (banned-word handling), hits_final, pass_step2,
clip_feat_path / clip_row, umt5_path / umt5_row (row of the per-dataset .npy).
pass_step2 = normalised ok AND n_cuts == 0 AND n_frames == 81 (2.1 / 2.2).

Usage: .venv-bpre/bin/python scripts/data_prep/bpre_index.py [--root runs/phaseAB/bpre]
"""
import argparse
import glob
import json
import os

import pandas as pd

DATASETS = ["vipe", "mixkit", "openvid"]


def jl(p):
    return [json.loads(l) for l in open(p)]


def ids_rows(p):
    return {c: i for i, c in enumerate(l.strip() for l in open(p) if l.strip())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/phaseAB/bpre")
    ap.add_argument("--mixkit_meta", default="data/mixkit_prep/meta.jsonl")
    a = ap.parse_args()
    R = a.root
    mk_meta = {m["id"]: m for m in jl(a.mixkit_meta)}
    frames = []
    for d in DATASETS:
        idx = pd.DataFrame(jl(f"{R}/clips/{d}_index.jsonl"))
        idx = idx[idx.status == "ok"]
        st = pd.DataFrame(jl(f"{R}/features/{d}_stats.jsonl"))
        cap_files = sorted(glob.glob(f"{R}/captions/{d}_s*of*.jsonl")) + sorted(glob.glob(f"{R}/captions/{d}_repair.jsonl"))
        caps = pd.DataFrame([r for p in cap_files for r in jl(p)])  # repair rows (if any) come last and win
        caps = caps.drop_duplicates("clip_id", keep="last")
        df = idx.merge(st, on="clip_id", how="left", suffixes=("", "_feat"))
        df = df.merge(caps[["clip_id", "caption_uni", "word_count_uni", "fix", "hits_final"]], on="clip_id", how="left")
        crow = ids_rows(f"{R}/features/{d}_clip_ids.txt")
        urow = ids_rows(f"{R}/umt5/{d}_uni_ids.txt")
        df["clip_feat_path"] = f"{R}/features/{d}_clip.npy"
        df["clip_row"] = df.clip_id.map(crow)
        df["umt5_path"] = f"{R}/umt5/{d}_uni.npy"
        df["umt5_row"] = df.clip_id.map(urow)
        df["word_count_orig"] = df.caption_orig.str.split().str.len()
        if d == "vipe":
            df["src_duration_s"] = 81 / 16  # HF ViPE `long` clips are the source (81 frames, 16 fps)
        elif d == "mixkit":
            sid = df.clip_id.str.slice(len("mixkit_"))
            df["src_duration_s"] = sid.map(lambda s: mk_meta[s]["n_src"] / mk_meta[s]["src_fps"])
        df["pass_step2"] = (df.n_cuts == 0) & (df.n_frames == 81)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    keep = ["clip_id", "dataset", "caption_orig", "caption_uni", "word_count_uni", "word_count_orig", "motion",
            "n_cuts", "n_frames", "src_duration_s", "fix", "hits_final", "pass_step2", "clip_feat_path", "clip_row",
            "umt5_path", "umt5_row", "src_shard", "src_key"]
    out = out[keep]
    miss = out[out.pass_step2 & (out.caption_uni.isna() | out.umt5_row.isna() | out.clip_row.isna())]
    print("pass_step2 rows missing caption/umt5/clip:", len(miss))
    out.to_parquet(f"{R}/index.parquet", index=False)
    print(out.groupby("dataset").agg(n=("clip_id", "size"), pass_step2=("pass_step2", "sum"),
                                     cuts=("n_cuts", lambda x: int((x > 0).sum()))))


if __name__ == "__main__":
    main()
