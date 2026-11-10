# SPDX-License-Identifier: Apache-2.0
"""VBench Quality / Semantic / Total Score with the official normalization and weights (task2.md B2.1). No torch.

Source: Vchitect/VBench scripts/constant.py (NORMALIZE_DIC, DIM_WEIGHT, QUALITY_WEIGHT = 4, SEMANTIC_WEIGHT = 1) and
scripts/cal_final_score.py, master at commit fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490 (fetched 2026-10-22):
    norm_d   = (s_d - min_d) / (max_d - min_d)
    quality  = sum_{d in quality}  w_d norm_d / sum w_d     (dynamic_degree w = 0.5, others 1)
    semantic = sum_{d in semantic} w_d norm_d / sum w_d     (all w = 1)
    total    = (4 quality + 1 semantic) / 5
The quality part is scripts/eval/vbench_quality_score.py (same constants).

Usage: python3 scripts/eval/vbench_total_score.py <quality eval_results.json> <semantic eval_results.json> <out.json>
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vbench_quality_score import quality_score  # noqa: E402

SEMANTIC_NORMALIZE = {
    "object_class": (0.0, 1.0),
    "multiple_objects": (0.0, 1.0),
    "human_action": (0.0, 1.0),
    "color": (0.0, 1.0),
    "spatial_relationship": (0.0, 1.0),
    "scene": (0.0, 0.8222),
    "appearance_style": (0.0009, 0.2855),
    "temporal_style": (0.0, 0.364),
    "overall_consistency": (0.0, 0.364),
}
SEMANTIC_WEIGHT_DIM = {d: 1.0 for d in SEMANTIC_NORMALIZE}
QUALITY_WEIGHT, SEMANTIC_WEIGHT = 4.0, 1.0


def semantic_score(results: dict) -> dict:
    raw = {d: float(results[d][0]) for d in SEMANTIC_NORMALIZE}
    norm = {d: (raw[d] - lo) / (hi - lo) for d, (lo, hi) in SEMANTIC_NORMALIZE.items()}
    s = sum(SEMANTIC_WEIGHT_DIM[d] * norm[d] for d in SEMANTIC_NORMALIZE) / sum(SEMANTIC_WEIGHT_DIM.values())
    n_videos = {d: len(results[d][1]) if len(results[d]) > 1 and isinstance(results[d][1], list) else None
                for d in SEMANTIC_NORMALIZE}
    return {"semantic_score": s, "raw": raw, "normalized": norm, "n_videos": n_videos}


def main():
    qpath, spath, out = sys.argv[1:4]
    q = quality_score(json.load(open(qpath)))
    s = semantic_score(json.load(open(spath)))
    total = (QUALITY_WEIGHT * q["quality_score"] + SEMANTIC_WEIGHT * s["semantic_score"]) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)
    res = {"total_score": total, "quality_score": q["quality_score"], "semantic_score": s["semantic_score"],
           "quality": q, "semantic": s, "inputs": [qpath, spath],
           "constants": "Vchitect/VBench scripts/constant.py @ fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"}
    json.dump(res, open(out, "w"), indent=1)
    print(f"total={total:.4f} quality={q['quality_score']:.4f} semantic={s['semantic_score']:.4f} -> {out}")


if __name__ == "__main__":
    main()
