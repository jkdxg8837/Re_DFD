"""B-pre 4.2 / 4.4 report for the chosen setting.

4.2 -> runs/phaseAB/bpre/clusters.html: per cluster of the chosen clustering (trimmed pool, K) the size per dataset,
       whether its cells are eligible, and 3 example unified captions per dataset.
4.4 -> runs/phaseAB/bpre/match/losses.json + markdown printed: per dataset the fraction of the post-step-2 pool that is
       not in the matched (eligible) pool, split into trimming / ineligible cells; and the top-10 dropped clusters
       (by number of dropped clips) of the full-pool UMT5 K = 128 clustering, with example captions.
"""
import argparse
import html
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bpre_match import DATASETS, M_MIN, load, make_cells  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/phaseAB/bpre")
    ap.add_argument("--tag", default="trim0.02_K128")
    a = ap.parse_args()
    R = a.root
    df, U, _ = load(R)
    info = json.load(open(f"{R}/splits/draw_info.json"))
    ids = [l.strip() for l in open(f"{R}/match/ids_{a.tag}.txt") if l.strip()]
    lab = np.load(f"{R}/match/labels_{a.tag}.npy")
    d = df.set_index("clip_id").loc[ids].reset_index()
    cells = make_cells(d, lab, info["motion_split"])
    d["cluster"], d["cell"] = lab, cells.values
    cnt = d.groupby(["cell", "dataset"]).size().unstack(fill_value=0).reindex(columns=DATASETS, fill_value=0)
    elig_cells = set(cnt.index[(cnt >= M_MIN).all(axis=1)])
    d["eligible"] = d.cell.isin(elig_cells)
    rng = np.random.default_rng(0)

    # 4.2 clusters.html
    rows = []
    for c, g in sorted(d.groupby("cluster"), key=lambda x: -len(x[1])):
        n = g.dataset.value_counts().reindex(DATASETS, fill_value=0)
        ne = g[g.eligible].dataset.value_counts().reindex(DATASETS, fill_value=0)
        ex = []
        for ds in DATASETS:
            gg = g[g.dataset == ds]
            pick = gg.caption_uni.values[rng.permutation(len(gg))[:3]]
            ex.append(f"<td>{'<hr>'.join(html.escape(x) for x in pick)}</td>")
        rows.append(f"<tr><td>{c}</td>" + "".join(f"<td>{n[ds]} ({ne[ds]})</td>" for ds in DATASETS) + "".join(ex) + "</tr>")
    page = ("<!doctype html><meta charset=utf-8><title>B-pre clusters</title><style>body{font:13px sans-serif}"
            "td{vertical-align:top;border-top:1px solid #ccc;padding:4px;max-width:420px}</style>"
            f"<h1>B-pre clusters ({a.tag}, motion split {info['motion_split']})</h1>"
            "<p>Size per dataset (in eligible cells); 3 random unified captions per dataset.</p><table>"
            "<tr><th>cluster</th><th>ViPE</th><th>Mixkit</th><th>OpenVid</th><th>ViPE examples</th>"
            "<th>Mixkit examples</th><th>OpenVid examples</th></tr>" + "".join(rows) + "</table>")
    open(f"{R}/clusters.html", "w").write(page)

    # 4.4 losses
    matched = set(d.clip_id[d.eligible])
    trimmed = set(d.clip_id)
    full = df.assign(cluster=np.load(f"{R}/match/labels_umt5_K128.npy"))
    full["status"] = np.where(full.clip_id.isin(matched), "matched",
                              np.where(full.clip_id.isin(trimmed), "ineligible_cell", "trimmed"))
    out = {"fractions": {}, "top_dropped_clusters": []}
    for ds in DATASETS:
        g = full[full.dataset == ds]
        out["fractions"][ds] = {k: round(float((g.status == k).mean()), 4) for k in ["matched", "trimmed", "ineligible_cell"]}
        out["fractions"][ds]["n_pool"] = int(len(g))
    drop = full[full.status != "matched"]
    for c, g in sorted(drop.groupby("cluster"), key=lambda x: -len(x[1]))[:10]:
        tot = full[full.cluster == c]
        e = dict(cluster=int(c), n_dropped=int(len(g)),
                 dropped_by_ds={ds: int((g.dataset == ds).sum()) for ds in DATASETS},
                 kept_frac_by_ds={ds: round(float((tot[tot.dataset == ds].status == "matched").mean()), 3)
                                  if (tot.dataset == ds).any() else None for ds in DATASETS},
                 examples=[f"[{r.dataset}] {r.caption_uni[:220]}" for r in g.sample(min(3, len(g)), random_state=0).itertuples()])
        out["top_dropped_clusters"].append(e)
    json.dump(out, open(f"{R}/match/losses.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
