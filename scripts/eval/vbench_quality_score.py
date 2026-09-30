# SPDX-License-Identifier: Apache-2.0
"""VBench Quality Score from a `<name>_eval_results.json`, using the official normalization and weights
(Vchitect/VBench `scripts/constant.py` + `scripts/cal_final_score.py`). No torch needed.

    quality = sum_d w_d * (s_d - min_d) / (max_d - min_d) / sum_d w_d   over the 7 quality dimensions

Usage: python3 scripts/eval/vbench_quality_score.py <eval_results.json> [more.json ...]
"""

import json
import os
import sys

# official constants (VBench scripts/constant.py)
NORMALIZE = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "temporal_flickering": (0.6293, 1.0),
    "motion_smoothness": (0.706, 0.9975),
    "dynamic_degree": (0.0, 1.0),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
}
WEIGHT = {d: 1.0 for d in NORMALIZE} | {"dynamic_degree": 0.5}


def quality_score(results: dict) -> dict:
    raw = {d: float(results[d][0]) for d in NORMALIZE}
    norm = {d: (raw[d] - lo) / (hi - lo) for d, (lo, hi) in NORMALIZE.items()}
    q = sum(WEIGHT[d] * norm[d] for d in NORMALIZE) / sum(WEIGHT.values())
    n_videos = {d: len(results[d][1]) if len(results[d]) > 1 and isinstance(results[d][1], list) else None
                for d in NORMALIZE}
    return {"quality_score": q, "raw": raw, "normalized": norm, "n_videos": n_videos}


def main():
    for path in sys.argv[1:]:
        out = quality_score(json.load(open(path)))
        dst = os.path.join(os.path.dirname(path), "quality_score.json")
        json.dump(out, open(dst, "w"), indent=1)
        print(f"{path}: quality_score={out['quality_score']:.4f} "
              f"motion_smoothness={out['raw']['motion_smoothness']:.4f} "
              f"dynamic_degree={out['raw']['dynamic_degree']:.4f} n_videos={out['n_videos']}")


if __name__ == "__main__":
    main()
