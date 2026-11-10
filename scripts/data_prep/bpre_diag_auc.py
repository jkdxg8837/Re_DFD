"""B-pre diagnostic (not in the spec): why the caption-classifier AUC stays high after cell matching, and what finer
matching would give. Writes runs/phaseAB/bpre/match/diag_auc.{json,md}.

1. baseline: random (unmatched) samples of N per dataset;
2. cell matching with finer K (256, 512; no motion split), dry-run sub1 as in the 5.1b table;
3. nearest-neighbour triplets: for each Mixkit clip (random order) the nearest unused ViPE and OpenVid clip by UMT5
   cosine (greedy, without replacement, faiss top-200); then keep the closest triplets (max of the two cosine
   distances below a quantile) and report AUC at several sizes;
4. most discriminative words per pair on the K = 32 (motion split) dry-run sub1 (log-odds with a +1 prior).
"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bpre_match import DATASETS, PAIRS, caption_auc, draw, kmeans, load, make_cells  # noqa: E402

ROOT = "runs/phaseAB/bpre"
OUT = f"{ROOT}/match"


def aucs(df, U, idx_by_ds):
    out = {}
    for a, b in PAIRS:
        ia, ib = idx_by_ds[a], idx_by_ds[b]
        X = U[np.r_[ia, ib]]
        y = np.r_[np.zeros(len(ia)), np.ones(len(ib))].astype(int)
        out[f"{a}-{b}"] = round(caption_auc(X, y), 3)
    return out


def main():
    import faiss

    df, U, C = load(ROOT)
    pos = {c: i for i, c in enumerate(df.clip_id)}
    by = {ds: np.where(df.dataset.values == ds)[0] for ds in DATASETS}
    rng = np.random.default_rng(0)
    res = {}
    N = 1100
    res["random_unmatched"] = aucs(df, U, {ds: rng.choice(by[ds], N, replace=False) for ds in DATASETS})
    print("random", res["random_unmatched"], flush=True)

    Uc = U - U.mean(0, keepdims=True)
    Uc /= np.linalg.norm(Uc, axis=1, keepdims=True)
    for emb_name, E, ks in (("raw", U, (256, 512)), ("centred", Uc, (32, 64, 128, 256))):
        for k in ks:
            lab = kmeans(E, k)
            np.save(f"{OUT}/diag_labels_umt5_{emb_name}_K{k}.npy", lab)  # never overwrite the table's labels
            roles, _, info = draw(df, make_cells(df, lab, False))
            idx = {ds: np.array([pos[c] for c in roles[(ds, "sub1")]]) for ds in DATASETS}
            r = dict(n_train=info["n_train"], eligible_pool=info["eligible_pool"], auc=aucs(df, U, idx))
            res[f"cells_{emb_name}_K{k}"] = r
            print(emb_name, k, r, flush=True)

    # nearest-neighbour triplets anchored on Mixkit
    res["nn_triplets"] = []
    for emb_name, E in (("raw", U), ("centred", Uc)):
        Uf = np.ascontiguousarray(E, dtype=np.float32)
        idxs = {}
        for ds in ("vipe", "openvid"):
            ix = faiss.IndexFlatIP(U.shape[1])
            ix.add(Uf[by[ds]])
            idxs[ds] = ix
        anchors = rng.permutation(by["mixkit"])
        used = {"vipe": set(), "openvid": set()}
        trip = []
        S = {ds: idxs[ds].search(Uf[anchors], 200) for ds in ("vipe", "openvid")}
        for j, m in enumerate(anchors):
            t = [m]
            dist = []
            for ds in ("vipe", "openvid"):
                sims, nn = S[ds][0][j], S[ds][1][j]
                pick = next(((s, by[ds][n]) for s, n in zip(sims, nn) if by[ds][n] not in used[ds]), None)
                if pick is None:
                    break
                used[ds].add(pick[1])
                t.append(pick[1])
                dist.append(1 - pick[0])
            if len(t) == 3:
                trip.append((t, max(dist)))
        trip.sort(key=lambda x: x[1])
        for n in (4800, 3000, 2000, 1100, 600):
            if n > len(trip):
                continue
            sel = trip[:n]
            idx = {ds: np.array([t[0][i] for t in sel]) for i, ds in enumerate(["mixkit", "vipe", "openvid"])}
            sub = {ds: rng.choice(idx[ds], min(N, n), replace=False) if n > N else idx[ds] for ds in DATASETS}
            r = dict(emb=emb_name, n_triplets=n, max_cos_dist=round(float(sel[-1][1]), 4), auc=aucs(df, U, sub))
            res["nn_triplets"].append(r)
            print("nn", r, flush=True)

    # discriminative words (K = 32 motion-split dry-run sub1)
    lab = np.load(f"{OUT}/labels_umt5_K32.npy")
    roles, _, _ = draw(df, make_cells(df, lab, True))
    cap = df.set_index("clip_id").caption_uni
    cnt = {ds: Counter(w for c in roles[(ds, "sub1")] for w in set(re.findall(r"[a-z]+", cap[c].lower())))
           for ds in DATASETS}
    n = {ds: len(roles[(ds, "sub1")]) for ds in DATASETS}
    res["words"] = {}
    for a, b in PAIRS:
        vocab = set(cnt[a]) | set(cnt[b])
        lo = {w: np.log((cnt[a][w] + 1) / (n[a] - cnt[a][w] + 1)) - np.log((cnt[b][w] + 1) / (n[b] - cnt[b][w] + 1))
              for w in vocab if cnt[a][w] + cnt[b][w] >= 30}
        top = sorted(lo, key=lo.get)
        fmt = lambda w: f"{w} ({cnt[a][w] / n[a]:.0%} vs {cnt[b][w] / n[b]:.0%})"  # noqa: E731
        res["words"][f"{a}-{b}"] = dict(more_in_a=[fmt(w) for w in top[::-1][:15]], more_in_b=[fmt(w) for w in top[:15]])
    json.dump(res, open(f"{OUT}/diag_auc.json", "w"), indent=1)
    print(json.dumps(res["words"], indent=1))


if __name__ == "__main__":
    main()
