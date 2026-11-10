"""B-pre step 6: balance diagnostics -> runs/phaseAB/bpre/BALANCE.{md,json}.

For each dataset pair and each comparison set:
  (a) full post-step-2 pools, (b) sub1, (c) R_* held-out sets, (d) R_<d> vs sub1_<d> within each dataset:
- TV distance between histograms on the reference clustering (UMT5 K_ref = 256), with a permutation baseline
  (dataset labels shuffled within the pooled pair, 200 times; target: TV <= 95th percentile);
- SMD (mean difference / sqrt of the mean of the two variances): caption word count (unified), motion,
  clip duration before trimming (report only);
- caption-classifier AUC (UMT5 pooled embeddings) and visual-classifier AUC (CLIP), as in bpre_match.caption_auc;
  for the full pools each dataset is subsampled to --pool_n clips (seed 0) for the classifiers only;
- banned-word hit rate of the unified captions per dataset.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bpre_caption import BANNED_RE  # noqa: E402
from bpre_match import DATASETS, PAIRS, caption_auc, load  # noqa: E402


def tv(a, b, k):
    pa = np.bincount(a, minlength=k) / len(a)
    pb = np.bincount(b, minlength=k) / len(b)
    return 0.5 * np.abs(pa - pb).sum()


def smd(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return float((x.mean() - y.mean()) / s) if s > 0 else 0.0


def compare(df, U, C, lab, ia, ib, k_ref, seed=0, n_perm=200, cls_n=None):
    rng = np.random.default_rng(seed)
    t = tv(lab[ia], lab[ib], k_ref)
    both = np.concatenate([lab[ia], lab[ib]])
    perm = []
    for _ in range(n_perm):
        p = rng.permutation(len(both))
        perm.append(tv(both[p[: len(ia)]], both[p[len(ia):]], k_ref))
    out = dict(n_a=int(len(ia)), n_b=int(len(ib)), tv=float(t), tv_perm_p95=float(np.percentile(perm, 95)),
               tv_perm_mean=float(np.mean(perm)))
    out["tv_ok"] = out["tv"] <= out["tv_perm_p95"]
    for col in ["word_count_uni", "motion", "src_duration_s"]:
        out[f"smd_{col}"] = smd(df[col].values[ia], df[col].values[ib])
    ca, cb = ia, ib
    if cls_n:
        ca = rng.choice(ia, min(cls_n, len(ia)), replace=False)
        cb = rng.choice(ib, min(cls_n, len(ib)), replace=False)
    idx = np.concatenate([ca, cb])
    y = np.r_[np.zeros(len(ca)), np.ones(len(cb))].astype(int)
    out["auc_caption"] = caption_auc(U[idx], y)
    out["auc_visual"] = caption_auc(C[idx], y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/phaseAB/bpre")
    ap.add_argument("--k_ref", type=int, default=256)
    ap.add_argument("--pool_n", type=int, default=3000)
    ap.add_argument("--splits_dir", default="splits")
    ap.add_argument("--out_name", default="BALANCE")
    a = ap.parse_args()
    df, U, C = load(a.root)
    lab = np.load(f"{a.root}/match/labels_umt5_K{a.k_ref}.npy")
    pos = {c: i for i, c in enumerate(df.clip_id)}

    def ids(ds, role):
        return np.array([pos[l.strip()] for l in open(f"{a.root}/{a.splits_dir}/{ds}_{role}.txt") if l.strip()])

    pools = {ds: np.where(df.dataset.values == ds)[0] for ds in DATASETS}
    res = {"banned_hit_rate": {}, "pairs": {}}
    for ds in DATASETS:
        for name, ix in [("pool", pools[ds]), ("sub1", ids(ds, "sub1")), ("ref", ids(ds, "ref"))]:
            caps = df.caption_uni.values[ix]
            res["banned_hit_rate"][f"{ds}_{name}"] = float(np.mean([bool(BANNED_RE.search(c)) for c in caps]))
    for setname in (["pool", "sub1", "ref"] if a.splits_dir == "splits" else ["sub1", "ref"]):
        for x, y in PAIRS:
            ia = pools[x] if setname == "pool" else ids(x, setname)
            ib = pools[y] if setname == "pool" else ids(y, setname)
            r = compare(df, U, C, lab, ia, ib, a.k_ref, cls_n=a.pool_n if setname == "pool" else None)
            res["pairs"][f"{setname}:{x}-{y}"] = r
            print(setname, x, y, json.dumps(r), flush=True)
    for ds in DATASETS:
        r = compare(df, U, C, lab, ids(ds, "ref"), ids(ds, "sub1"), a.k_ref)
        res["pairs"][f"ref-vs-sub1:{ds}"] = r
        print("ref-vs-sub1", ds, json.dumps(r), flush=True)
    json.dump(res, open(f"{a.root}/{a.out_name}.json", "w"), indent=1)

    def ok(v, thr):
        return "✓" if abs(v) < thr else "✗"

    L = ["# B-pre balance diagnostics", "",
         f"Reference clustering: UMT5 spherical K-means K = {a.k_ref} (not the matching clustering). "
         f"Full pools: classifiers on {a.pool_n} random clips per dataset. Targets apply to `sub1`.", "",
         "| set | pair | n | TV (perm. p95) | SMD words | SMD motion | SMD duration (report) | caption AUC | visual AUC (report) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for key, r in res["pairs"].items():
        s, p = key.split(":")
        tgt = s == "sub1"
        L.append(f"| {s} | {p} | {r['n_a']} / {r['n_b']} | {r['tv']:.3f} ({r['tv_perm_p95']:.3f})"
                 f"{' ' + ('✓' if r['tv_ok'] else '✗') if tgt else ''} | {r['smd_word_count_uni']:+.3f}"
                 f"{' ' + ok(r['smd_word_count_uni'], 0.1) if tgt else ''} | {r['smd_motion']:+.3f}"
                 f"{' ' + ok(r['smd_motion'], 0.1) if tgt else ''} | {r['smd_src_duration_s']:+.3f} | "
                 f"{r['auc_caption']:.3f}{' ' + ('✓' if r['auc_caption'] < 0.6 else '✗') if tgt else ''} | "
                 f"{r['auc_visual']:.3f} |")
    L += ["", "Banned-word hit rate (unified captions): "
          + ", ".join(f"{k} {v:.4f}" for k, v in res["banned_hit_rate"].items())]
    open(f"{a.root}/{a.out_name}.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
