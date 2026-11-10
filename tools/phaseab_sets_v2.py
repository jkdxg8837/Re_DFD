# SPDX-License-Identifier: Apache-2.0
"""Phase 0.1 / 0.2 redo with the content-matched B-pre sets (data_prep.md amendments A-1 / A-2).

  R_mx / R_ov / S_vipe = runs/phaseAB/bpre/splits/{mixkit,openvid,vipe}_ref.txt (300 each, identical cell histogram),
                         clips = the B-pre normalised clips (832x480, 16 fps, 81 frames; the same files the
                         `_uni` training shards contain), caption = unified Qwen3-VL-4B caption
  prompts              = runs/phaseAB/bpre/splits/eval_prompts.jsonl (200, one unified caption per cell, source
                         rotating vipe / mixkit / openvid); seed = 1000 + prompt_idx (unchanged rule)
Writes <PHASEAB_ROOT>/sets/<set>.jsonl {"idx", "id", "mp4", "caption"} and <PHASEAB_ROOT>/prompts.jsonl
{"prompt_idx", "source", "src_id", "prompt", "seed"} (same schema as tools/phaseab_sets.py). No clip copies.
Usage: PHASEAB_ROOT=runs/phaseAB/v2 .venv-bpre/bin/python tools/phaseab_sets_v2.py
"""

import json
import os

import pandas as pd

ROOT = os.environ.get("PHASEAB_ROOT", "runs/phaseAB/v2")
BPRE = "runs/phaseAB/bpre"
SETS = {"R_mx": "mixkit", "R_ov": "openvid", "S_vipe": "vipe"}


def main():
    assert ROOT != "runs/phaseAB", "never overwrite the first Phase 0 / A sets"
    os.makedirs(os.path.join(ROOT, "sets"), exist_ok=True)
    cap = pd.read_parquet(f"{BPRE}/index.parquet").set_index("clip_id").caption_uni
    for name, ds in SETS.items():
        ids = [l.strip() for l in open(f"{BPRE}/splits/{ds}_ref.txt") if l.strip()]
        assert len(ids) == 300
        with open(os.path.join(ROOT, "sets", f"{name}.jsonl"), "w") as f:
            for i, c in enumerate(ids):
                mp4 = f"{BPRE}/clips/{ds}/{c}.mp4"
                assert os.path.exists(mp4), mp4
                f.write(json.dumps({"idx": i, "id": c, "mp4": mp4, "caption": cap[c]}) + "\n")
        print(name, len(ids))
    ev = [json.loads(l) for l in open(f"{BPRE}/splits/eval_prompts.jsonl")]
    assert [e["prompt_idx"] for e in ev] == list(range(200))
    with open(os.path.join(ROOT, "prompts.jsonl"), "w") as f:
        for e in ev:
            f.write(json.dumps({"prompt_idx": e["prompt_idx"], "source": e["source"], "src_id": e["clip_id"],
                                "prompt": e["prompt"], "seed": 1000 + e["prompt_idx"]}) + "\n")
    print("prompts", len(ev), {s: sum(e["source"] == s for e in ev) for s in SETS.values()})


if __name__ == "__main__":
    main()
