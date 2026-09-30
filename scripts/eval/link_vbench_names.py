# SPDX-License-Identifier: Apache-2.0
"""Expose inference outputs under VBench's standard file names `<prompt>-<k>.mp4` via relative symlinks.

`video_model_inference.py` writes `<raw>/seed<k>/<prompt_file_stem>/student_step4_<i:04d>_seed<k>.mp4`, where i is
the line index in the prompt file. Fails loudly if any of the len(prompts) x 5 videos is missing.
"""

import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=5)
    a = ap.parse_args()
    prompts = [line.rstrip("\n") for line in open(a.prompts) if line.strip()]
    os.makedirs(a.out, exist_ok=True)
    missing = []
    for k in range(a.samples):
        for i, p in enumerate(prompts):
            hits = glob.glob(os.path.join(a.raw, f"seed{k}", "*", f"student_step*_{i:04d}_seed{k}.mp4"))
            if len(hits) != 1:
                missing.append((k, i, len(hits)))
                continue
            dst = os.path.join(a.out, f"{p}-{k}.mp4")
            if os.path.islink(dst):
                os.unlink(dst)  # replace a symlink created by an earlier run
            elif os.path.exists(dst):
                raise SystemExit(f"{dst} exists and is not a symlink; refusing to touch it")
            os.symlink(os.path.relpath(hits[0], a.out), dst)
    n = len(prompts) * a.samples
    print(f"linked {n - len(missing)}/{n} videos into {a.out}")
    if missing:
        raise SystemExit(f"missing or ambiguous (seed, prompt_idx, hits): {missing[:20]}")


if __name__ == "__main__":
    main()
