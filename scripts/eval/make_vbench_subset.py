# SPDX-License-Identifier: Apache-2.0
"""Build the fixed VBench subset used for every model (spec §6, decision of 2026-09-24).

200 videos per Quality-Score dimension = 40 official prompts x 5 samples. The 7 quality dimensions use 4 disjoint
official prompt sets, so prompts are sampled once per set and shared by the dimensions of that set:

    {subject_consistency, motion_smoothness, dynamic_degree}: 72 prompts
    {background_consistency}: 86
    {temporal_flickering}: 75
    {aesthetic_quality, imaging_quality}: 93

-> 160 prompts x 5 samples = 800 videos per evaluated model.

Outputs (in --out):
    prompts.txt           one prompt per line, fixed order (input of scripts/inference/video_model_inference.py)
    subset_full_info.json VBench full_info restricted to the chosen prompts and the 7 quality dimensions
    subset_meta.json      seed, per-dimension prompt counts, source file checksum
"""

import argparse
import hashlib
import json
import os
import random

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
    ap.add_argument("--full_info", default="runs/eval/vbench_subset/VBench_full_info_official.json")
    ap.add_argument("--out", default="runs/eval/vbench_subset")
    ap.add_argument("--prompts_per_dim", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    raw = open(a.full_info, "rb").read()
    info = json.loads(raw)
    # prompt -> set of quality dimensions it is officially used for (merges the few duplicated entries)
    dims_of = {}
    for e in info:
        q = [d for d in e["dimension"] if d in QUALITY_DIMS]
        if q:
            dims_of.setdefault(e["prompt_en"], set()).update(q)

    # group dimensions that share exactly the same official prompt set
    groups = {}
    for d in QUALITY_DIMS:
        groups.setdefault(frozenset(p for p, ds in dims_of.items() if d in ds), []).append(d)
    rng = random.Random(a.seed)
    chosen = {}  # prompt -> dims
    for prompt_set, dims in sorted(groups.items(), key=lambda kv: QUALITY_DIMS.index(kv[1][0])):
        assert not (prompt_set & chosen.keys()), "quality prompt sets are expected to be disjoint"
        for p in rng.sample(sorted(prompt_set), a.prompts_per_dim):
            chosen[p] = sorted(dims_of[p] & set(dims), key=QUALITY_DIMS.index)

    prompts = list(chosen)
    assert all("\n" not in p and "/" not in p for p in prompts), "prompts are used as file names"
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "prompts.txt"), "w") as f:
        f.write("\n".join(prompts) + "\n")
    with open(os.path.join(a.out, "subset_full_info.json"), "w") as f:
        json.dump([{"prompt_en": p, "dimension": chosen[p]} for p in prompts], f, indent=1)
    per_dim = {d: sum(d in ds for ds in chosen.values()) for d in QUALITY_DIMS}
    meta = {
        "source": a.full_info,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "seed": a.seed,
        "prompts_per_dim": a.prompts_per_dim,
        "samples_per_prompt": 5,
        "num_prompts": len(prompts),
        "prompts_per_dimension": per_dim,
        "videos_per_dimension": {d: n * 5 for d, n in per_dim.items()},
        "groups": [dims for dims in groups.values()],
    }
    with open(os.path.join(a.out, "subset_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
