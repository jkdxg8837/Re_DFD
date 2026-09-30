# SPDX-License-Identifier: Apache-2.0
"""Run VBench (standard mode) on the fixed subset for the 7 Quality-Score dimensions. Run inside `.venv-vbench`.

Each dimension is evaluated and saved separately (`<name>_<dim>_eval_results.json`), and finished dimensions are
skipped on re-runs. The merged result is written to `<name>_eval_results.json`.

Usage:
    VBENCH_CACHE_DIR=data/vbench_cache .venv-vbench/bin/python scripts/eval/run_vbench.py \
        --videos runs/eval/<tag>/videos --name <tag> --out runs/eval/<tag>/vbench
Then: python3 scripts/eval/vbench_quality_score.py runs/eval/<tag>/vbench/<tag>_eval_results.json
"""

import argparse
import json
import os

# VBench 0.1.5 loads its (official, trusted) model weights with plain torch.load; torch >= 2.6 defaults to
# weights_only=True and rejects e.g. the AMT checkpoint used by motion_smoothness.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

QUALITY_DIMS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full_info", default="runs/eval/vbench_subset/subset_full_info.json")
    ap.add_argument("--dims", nargs="+", default=QUALITY_DIMS)
    a = ap.parse_args()

    import torch
    from vbench import VBench

    n = len([f for f in os.listdir(a.videos) if f.endswith(".mp4")])
    print(f"{n} videos in {a.videos}; dims: {a.dims}", flush=True)
    bench = VBench(torch.device("cuda"), a.full_info, a.out)
    merged = {}
    for dim in a.dims:
        path = os.path.join(a.out, f"{a.name}_{dim}_eval_results.json")
        if not os.path.exists(path):
            print(f"=== evaluating {dim}", flush=True)
            bench.evaluate(videos_path=a.videos, name=f"{a.name}_{dim}", dimension_list=[dim], mode="vbench_standard")
        merged.update(json.load(open(path)))
        print(f"=== {dim}: {merged[dim][0]}", flush=True)
    with open(os.path.join(a.out, f"{a.name}_eval_results.json"), "w") as f:
        json.dump(merged, f)


if __name__ == "__main__":
    main()
