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

--with_semantic (task2.md B2.1, default off): also the 9 Semantic-Score dimensions, with the same rule applied to
all 16 dimensions in order (quality first): dimensions with identical official prompt sets form one group that is
sampled once (40 prompts), so the quality part (prompts, order, dims) is unchanged; scene shares the
background_consistency prompts and overall_consistency the aesthetic/imaging prompts. The semantic-only groups may
overlap a quality prompt set (appearance_style shares 2 prompts with aesthetic/imaging): a prompt drawn twice keeps
one video set and gets the dims of both draws. Extra outputs: prompts_new.txt (prompts not in the quality subset,
to be generated) and the official auxiliary_info of the semantic dims in subset_full_info.json.
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
SEMANTIC_DIMS = [
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_info", default="runs/eval/vbench_subset/VBench_full_info_official.json")
    ap.add_argument("--out", default="runs/eval/vbench_subset")
    ap.add_argument("--prompts_per_dim", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with_semantic", action="store_true", help="also the 9 semantic dimensions (see docstring)")
    a = ap.parse_args()
    if a.with_semantic:
        return main_with_semantic(a)

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


def main_with_semantic(a):
    raw = open(a.full_info, "rb").read()
    info = json.loads(raw)
    dims_all = QUALITY_DIMS + SEMANTIC_DIMS
    dims_of, aux_of = {}, {}
    for e in info:
        q = [d for d in e["dimension"] if d in dims_all]
        if q:
            dims_of.setdefault(e["prompt_en"], set()).update(q)
            for d, v in (e.get("auxiliary_info") or {}).items():
                aux_of.setdefault(e["prompt_en"], {})[d] = v
    groups = {}
    for d in dims_all:
        groups.setdefault(frozenset(p for p, ds in dims_of.items() if d in ds), []).append(d)
    rng = random.Random(a.seed)
    chosen = {}  # prompt -> dims (insertion order = prompt order)
    for prompt_set, dims in sorted(groups.items(), key=lambda kv: dims_all.index(kv[1][0])):
        quality_group = dims[0] in QUALITY_DIMS
        if quality_group:
            assert not (prompt_set & chosen.keys()), "quality prompt sets are expected to be disjoint"
        for p in rng.sample(sorted(prompt_set), a.prompts_per_dim):
            new = dims_of[p] & set(dims)
            chosen[p] = sorted(set(chosen.get(p, [])) | new, key=dims_all.index)

    prompts = list(chosen)
    assert all("\n" not in p and "/" not in p for p in prompts), "prompts are used as file names"
    # the quality part must equal the existing quality-only subset
    q_ref = os.path.join(os.path.dirname(a.full_info), "prompts.txt")
    q_prompts = [p for p in prompts if set(chosen[p]) & set(QUALITY_DIMS)]
    if os.path.exists(q_ref):
        ref = [l.rstrip("\n") for l in open(q_ref) if l.strip()]
        assert q_prompts == ref == prompts[: len(ref)], "quality part differs from the existing subset"
        ref_info = {e["prompt_en"]: e["dimension"] for e in json.load(open(os.path.join(os.path.dirname(a.full_info),
                                                                                      "subset_full_info.json")))}
        assert all([d for d in chosen[p] if d in QUALITY_DIMS] == ref_info[p] for p in ref), "quality dims differ"
    new_prompts = prompts[len(q_prompts):]
    assert not (set(new_prompts) & set(q_prompts))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "prompts.txt"), "w") as f:
        f.write("\n".join(prompts) + "\n")
    with open(os.path.join(a.out, "prompts_new.txt"), "w") as f:
        f.write("\n".join(new_prompts) + "\n")
    entries = []
    for p in prompts:
        e = {"prompt_en": p, "dimension": chosen[p]}
        aux = {d: v for d, v in aux_of.get(p, {}).items() if d in chosen[p]}
        if aux:
            e["auxiliary_info"] = aux
        entries.append(e)
    with open(os.path.join(a.out, "subset_full_info.json"), "w") as f:
        json.dump(entries, f, indent=1)
    per_dim = {d: sum(d in ds for ds in chosen.values()) for d in dims_all}
    meta = {
        "source": a.full_info,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "seed": a.seed,
        "prompts_per_dim": a.prompts_per_dim,
        "samples_per_prompt": 5,
        "num_prompts": len(prompts),
        "num_prompts_quality_subset": len(q_prompts),
        "num_prompts_new": len(new_prompts),
        "prompts_per_dimension": per_dim,
        "videos_per_dimension": {d: n * 5 for d, n in per_dim.items()},
        "prompts_shared_by_two_draws": [p for p in prompts if len({tuple(g) for g in groups.values()
                                                                   if set(g) & set(chosen[p])}) > 1],
        "groups": [dims for dims in groups.values()],
    }
    with open(os.path.join(a.out, "subset_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
