"""B-pre steps 4 (joint clustering) and 5 (cells, eligibility, choice of K, stratified draws).

  table : 4.1 FAISS spherical K-means on pooled UMT5 caption embeddings for K in {16,32,64,128} + K_ref = 256,
          4.3 CLIP K-means (K = 256) purity, and the 5.1b table (per K, with / without motion split:
          eligible pool per dataset, max N_train, caption-classifier AUC on a dry-run sub1) -> <out>/table.{json,md}
  draw  : 5.2 final draws for one setting (--k, --motion_split) -> <root>/splits/<dataset>_<role>.txt

Draw (identical cell histogram for every dataset in every role; roles disjoint within a dataset):
  cell = (cluster, motion bin), motion bin = motion > per-cluster median of the pooled three datasets;
  eligible if every dataset has >= m = 2 clips. cap_c = min over datasets of the remaining clips in cell c.
  1. held-out refs (300 each): allocation over cells proportional to cap_c (largest remainder, <= cap_c).
  2. eval prompts (200): cells drawn proportional to cap_c (without replacement while possible), source dataset
     rotating vipe / mixkit / openvid; one random clip of that dataset in that cell gives the caption and is removed.
  3. train sub1 / sub2 / sub3 (seeds 101 / 102 / 103), disjoint: a_c = floor(cap_c / 3) per subset at the maximum
     size, scaled down (largest remainder) for a smaller N_train.
Caption-classifier AUC: standardise + L2 logistic regression (C chosen by inner 3-fold CV over 1e-4..1),
5-fold stratified outer CV, ROC AUC, per dataset pair.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

DATASETS = ["vipe", "mixkit", "openvid"]
PAIRS = [("vipe", "mixkit"), ("vipe", "openvid"), ("mixkit", "openvid")]
KS = [16, 32, 64, 128]
K_REF = 256
M_MIN = 2


def load(root):
    df = pd.read_parquet(f"{root}/index.parquet")
    df = df[df.pass_step2].reset_index(drop=True)

    def gather(path_col, row_col):
        out = np.zeros((len(df), 0), np.float32)
        parts = {}
        for p, g in df.groupby(path_col):
            arr = np.load(p, mmap_mode="r")
            parts[p] = (g.index.values, np.asarray(arr[g[row_col].astype(int).values]))
        dim = next(iter(parts.values()))[1].shape[1]
        out = np.zeros((len(df), dim), np.float32)
        for ix, v in parts.values():
            out[ix] = v
        return out

    return df, gather("umt5_path", "umt5_row"), gather("clip_feat_path", "clip_row")


def kmeans(x, k, seed=0):
    import faiss

    km = faiss.Kmeans(x.shape[1], k, niter=50, seed=seed, spherical=True, verbose=False)
    km.train(np.ascontiguousarray(x, dtype=np.float32))
    _, lab = km.index.search(np.ascontiguousarray(x, dtype=np.float32), 1)
    return lab[:, 0]


def make_cells(df, labels, motion_split):
    cl = pd.Series(labels, index=df.index)
    if not motion_split:
        return cl.astype(str)
    med = df.motion.groupby(cl).transform("median")
    return cl.astype(str) + "_" + (df.motion > med).astype(int).astype(str)


def largest_remainder(weights, total, caps):
    w = np.asarray(weights, float)
    caps = np.asarray(caps, int)
    total = int(min(total, caps.sum()))
    alloc = np.zeros(len(w), int)
    while alloc.sum() < total:
        free = alloc < caps
        rest = total - alloc.sum()
        q = np.where(free, w, 0)
        q = q / q.sum() * rest
        add = np.minimum(np.floor(q).astype(int), caps - alloc)
        if add.sum() == 0:  # hand out the remainder one by one by largest fractional part
            order = np.argsort(-(q - np.floor(q)) - 1e-9 * np.arange(len(q)))
            for i in order:
                if alloc.sum() >= total:
                    break
                if alloc[i] < caps[i] and q[i] > 0:
                    alloc[i] += 1
        alloc += add
    return alloc


def draw(df, cells, n_ref=300, n_eval=200, n_train=None, seeds=(101, 102, 103), seed_ref=0, seed_eval=1,
         train_roles=True):
    """Returns (roles: {(dataset, role): [clip_id]}, eval_prompts: [dict], info)."""
    d = df.assign(cell=cells.values)
    cnt = d.groupby(["cell", "dataset"]).size().unstack(fill_value=0).reindex(columns=DATASETS, fill_value=0)
    elig = cnt.index[(cnt >= M_MIN).all(axis=1)]
    pool = {(c, ds): list(g.clip_id) for (c, ds), g in d[d.cell.isin(elig)].groupby(["cell", "dataset"])}
    rng0 = np.random.default_rng(seed_ref)
    for k in pool:
        rng0.shuffle(pool[k])  # one fixed random order per (cell, dataset); roles take from the front
    cells_e = list(elig)

    def cap():
        return np.array([min(len(pool[(c, ds)]) for ds in DATASETS) for c in cells_e])

    def take(role_counts, role):
        out = {ds: [] for ds in DATASETS}
        for c, n in zip(cells_e, role_counts):
            for ds in DATASETS:
                out[ds] += pool[(c, ds)][:n]
                pool[(c, ds)] = pool[(c, ds)][n:]
        return {(ds, role): v for ds, v in out.items()}

    roles = {}
    c0 = cap()
    info = dict(n_cells=int(len(cnt)), n_eligible=int(len(elig)),
                eligible_pool={ds: int(cnt.loc[elig, ds].sum()) for ds in DATASETS},
                total_pool={ds: int(cnt[ds].sum()) for ds in DATASETS}, matched_capacity=int(c0.sum()))
    # 1. held-out reference sets
    roles.update(take(largest_remainder(c0, n_ref, c0), "ref"))
    # 2. eval prompt pool
    rng = np.random.default_rng(seed_eval)
    prompts, used = [], set()
    w = cap().astype(float)  # matched cell size after the reference draw
    while len(prompts) < n_eval:
        ds = DATASETS[len(prompts) % 3]
        ok = np.array([len(pool[(c, ds)]) > 0 for c in cells_e])
        cand = ok & np.array([c not in used for c in cells_e])
        if not cand.any():  # every cell used once -> sample with replacement
            cand = ok
        if not cand.any():
            break
        p = np.where(cand, w, 0.0)
        p = p if p.sum() > 0 else cand.astype(float)
        c = cells_e[rng.choice(len(cells_e), p=p / p.sum())]
        used.add(c)
        prompts.append(dict(prompt_idx=len(prompts), cell=c, source=ds, clip_id=pool[(c, ds)].pop(0)))
    roles.update({(ds, "eval"): [p["clip_id"] for p in prompts if p["source"] == ds] for ds in DATASETS})
    info["n_eval"] = len(prompts)
    info["n_eval_distinct_cells"] = len(used)
    # 3. training subsets
    c2 = cap()
    a_max = c2 // 3
    info["max_n_train_disjoint"] = int(a_max.sum())
    info["max_n_train_single"] = int(c2.sum())
    if train_roles:
        a = a_max if n_train is None else largest_remainder(a_max, n_train, a_max)
        for j, s in enumerate(seeds):
            r = np.random.default_rng(s)
            out = {ds: [] for ds in DATASETS}
            for c, n in zip(cells_e, a):
                for ds in DATASETS:
                    lst = pool[(c, ds)]
                    pick = sorted(r.choice(len(lst), size=n, replace=False).tolist()) if n else []
                    out[ds] += [lst[i] for i in pick]
                    pool[(c, ds)] = [x for i, x in enumerate(lst) if i not in set(pick)]
            roles.update({(ds, f"sub{j + 1}"): v for ds, v in out.items()})
        info["n_train"] = int(a.sum())
    return roles, prompts, info


def caption_auc(X, y, seed=0):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], cv=3,
                                                               scoring="roc_auc", max_iter=2000))
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba", n_jobs=5)[:, 1]
    return float(roc_auc_score(y, p))


def pair_aucs(df, X, ids):
    sel = df.clip_id.isin(ids).values
    out = {}
    for a, b in PAIRS:
        m = sel & df.dataset.isin([a, b]).values
        out[f"{a}-{b}"] = caption_auc(X[m], (df.dataset.values[m] == b).astype(int))
    return out


def cmd_table(a):
    df, U, C = load(a.root)
    os.makedirs(a.out, exist_ok=True)
    print(df.dataset.value_counts().to_dict(), flush=True)
    res = dict(pool=df.dataset.value_counts().to_dict(), settings=[])
    for k in KS + [K_REF]:
        lab = kmeans(U, k)
        np.save(f"{a.out}/labels_umt5_K{k}.npy", lab)
    clip_lab = kmeans(C, 256)
    np.save(f"{a.out}/labels_clip_K256.npy", clip_lab)
    t = pd.crosstab(clip_lab, df.dataset.values)
    res["clip_purity_K256"] = float((t.max(axis=1)).sum() / t.values.sum())
    res["clip_purity_K256_by_majority"] = {ds: int((t.idxmax(axis=1) == ds).sum()) for ds in DATASETS}
    pd.Series(df.clip_id).to_csv(f"{a.out}/clip_ids.csv", index=False)
    for k in KS:
        lab = np.load(f"{a.out}/labels_umt5_K{k}.npy")
        for ms in (True, False):
            cells = make_cells(df, lab, ms)
            roles, prompts, info = draw(df, cells)
            sub1 = [c for ds in DATASETS for c in roles[(ds, "sub1")]]
            row = dict(K=k, motion_split=ms, **info)
            row["dropped_frac"] = {ds: 1 - info["eligible_pool"][ds] / info["total_pool"][ds] for ds in DATASETS}
            row["auc_sub1"] = pair_aucs(df, U, sub1) if info["n_train"] >= 20 else None
            res["settings"].append(row)
            print(json.dumps(row), flush=True)
            json.dump(res, open(f"{a.out}/table.json", "w"), indent=1)
    lines = ["| K | motion split | eligible cells / cells | eligible pool ViPE / MX / OV | dropped ViPE / MX / OV | "
             "max N_train (disjoint) | AUC V-M / V-O / M-O (dry-run sub1) |", "|---|---|---|---|---|---|---|"]
    for r in res["settings"]:
        e, dr, au = r["eligible_pool"], r["dropped_frac"], r["auc_sub1"] or {}
        lines.append(f"| {r['K']} | {'yes' if r['motion_split'] else 'no'} | {r['n_eligible']} / {r['n_cells']} | "
                     f"{e['vipe']} / {e['mixkit']} / {e['openvid']} | "
                     f"{dr['vipe']:.0%} / {dr['mixkit']:.0%} / {dr['openvid']:.0%} | {r['max_n_train_disjoint']} | "
                     + " / ".join(f"{au.get(p, float('nan')):.3f}" for p in ["vipe-mixkit", "vipe-openvid", "mixkit-openvid"]) + " |")
    lines.append(f"\nCLIP K-means (K = 256) size-weighted purity: {res['clip_purity_K256']:.3f}")
    open(f"{a.out}/table.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def write_draw(root, d, roles, prompts, info, splits="splits"):
    os.makedirs(f"{root}/{splits}", exist_ok=True)
    for (ds, role), ids in roles.items():
        open(f"{root}/{splits}/{ds}_{role}.txt", "w").write("\n".join(ids) + "\n")
    cap = d.set_index("clip_id").caption_uni
    with open(f"{root}/{splits}/eval_prompts.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps(dict(**p, prompt=cap[p["clip_id"]])) + "\n")
    info["sizes"] = {f"{ds}_{r}": len(v) for (ds, r), v in roles.items()}
    json.dump(info, open(f"{root}/{splits}/draw_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


def cmd_select(a):
    """Chosen setting (user, 2026-10-24): propensity trimming (min_d p(d | caption) >= tau, 5-fold cross-fitted
    3-class logistic regression, match/diag_propensity.npy) + cell matching with K on the trimmed pool; motion split
    kept if its max N_train is >= 85 % of the no-split one (5.1b rule); then the 5.2 draws at the max disjoint size."""
    df, U, C = load(a.root)
    P = np.load(f"{a.out}/diag_propensity.npy")
    assert len(P) == len(df)
    excl = set(x for x in a.exclude.split(",") if x)
    keep = (P.min(1) >= a.trim_tau) & ~df.clip_id.isin(excl).values
    d = df[keep].reset_index(drop=True)
    Uk = U[keep]
    lab = kmeans(Uk, a.k)
    tag = f"trim{a.trim_tau}_K{a.k}" + ("" if a.splits_dir == "splits" else f"_{a.splits_dir}")
    np.save(f"{a.out}/labels_{tag}.npy", lab)
    open(f"{a.out}/ids_{tag}.txt", "w").write("\n".join(d.clip_id) + "\n")
    res = dict(trim_tau=a.trim_tau, K=a.k, excluded=sorted(excl), n_before=df.dataset.value_counts().to_dict(),
               n_trimmed=d.dataset.value_counts().to_dict(), settings={})
    for ms in (True, False):
        roles, prompts, info = draw(d, make_cells(d, lab, ms))
        sub1 = [c for ds in DATASETS for c in roles[(ds, "sub1")]]
        info["auc_sub1"] = pair_aucs(d, Uk, sub1)
        res["settings"]["split" if ms else "nosplit"] = info
        print(ms, json.dumps(info), flush=True)
    ms = res["settings"]["split"]["max_n_train_disjoint"] >= 0.85 * res["settings"]["nosplit"]["max_n_train_disjoint"]
    if a.motion_split_mode != "auto":  # explicit override (e.g. motion SMD failed in step 6)
        ms = a.motion_split_mode == "1"
    res["motion_split_chosen"] = bool(ms)
    json.dump(res, open(f"{a.out}/select_{tag}.json", "w"), indent=1)
    roles, prompts, info = draw(d, make_cells(d, lab, ms), n_train=a.n_train)
    info.update(trim_tau=a.trim_tau, k=a.k, motion_split=bool(ms), excluded=sorted(excl),
                auc_sub1_dryrun=res["settings"]["split" if ms else "nosplit"]["auc_sub1"])
    write_draw(a.root, d, roles, prompts, info, a.splits_dir)


def cmd_draw(a):
    df, U, C = load(a.root)
    lab = np.load(f"{a.root}/match/labels_umt5_K{a.k}.npy")
    cells = make_cells(df, lab, a.motion_split)
    roles, prompts, info = draw(df, cells, n_train=a.n_train)
    os.makedirs(f"{a.root}/splits", exist_ok=True)
    for (ds, role), ids in roles.items():
        open(f"{a.root}/splits/{ds}_{role}.txt", "w").write("\n".join(ids) + "\n")
    cap = df.set_index("clip_id").caption_uni
    with open(f"{a.root}/splits/eval_prompts.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps(dict(**p, prompt=cap[p["clip_id"]])) + "\n")
    info.update(k=a.k, motion_split=a.motion_split, sizes={f"{ds}_{r}": len(v) for (ds, r), v in roles.items()})
    json.dump(info, open(f"{a.root}/splits/draw_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table", "draw", "select"])
    ap.add_argument("--root", default="runs/phaseAB/bpre")
    ap.add_argument("--out", default="runs/phaseAB/bpre/match")
    ap.add_argument("--k", type=int)
    ap.add_argument("--motion_split", type=int, default=1)
    ap.add_argument("--n_train", type=int, default=None)
    ap.add_argument("--trim_tau", type=float, default=0.02)
    ap.add_argument("--motion_split_mode", choices=["auto", "0", "1"], default="auto")
    ap.add_argument("--splits_dir", default="splits")
    ap.add_argument("--exclude", default="vipe_00010270", help="clip ids removed from the pool (comma-separated)")
    a = ap.parse_args()
    {"table": cmd_table, "draw": cmd_draw, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
