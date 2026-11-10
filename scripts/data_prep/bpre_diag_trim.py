"""B-pre diagnostic (not in the spec): propensity trimming before cell matching.

3-class logistic regression (standardise + L2, C = 0.01) on the UMT5 caption embeddings of the whole post-step-2 pool,
5-fold cross-fitted probabilities p(dataset | caption) for every clip; keep clips with min_d p(d | x) >= tau (the
caption is plausible under every dataset); then cell matching (K, no motion split) on the trimmed pool and the dry-run
sub1 caption-classifier AUC as in the 5.1b table. Writes runs/phaseAB/bpre/match/diag_trim.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bpre_match import DATASETS, PAIRS, caption_auc, draw, kmeans, load, make_cells  # noqa: E402

ROOT = "runs/phaseAB/bpre"


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df, U, _ = load(ROOT)
    y = df.dataset.map({d: i for i, d in enumerate(DATASETS)}).values
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced"))
    pf = f"{ROOT}/match/diag_propensity.npy"
    if os.path.exists(pf):
        P = np.load(pf)
    else:
        P = cross_val_predict(clf, U, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba", n_jobs=5)
        np.save(pf, P)
    pmin = P.min(1)
    pos = {c: i for i, c in enumerate(df.clip_id)}
    res = []
    for tau in [float(t) for t in os.environ.get('TAUS', '0.0,0.05').split(',')]:
        keep = np.where(pmin >= tau)[0]
        d = df.iloc[keep].reset_index(drop=True)
        for k in (32, 64, 128):
            lab = kmeans(U[keep], k)
            roles, _, info = draw(d, make_cells(d, lab, False))
            auc = {}
            for a, b in PAIRS:
                ia = np.array([pos[c] for c in roles[(a, "sub1")]])
                ib = np.array([pos[c] for c in roles[(b, "sub1")]])
                auc[f"{a}-{b}"] = round(caption_auc(U[np.r_[ia, ib]], np.r_[np.zeros(len(ia)), np.ones(len(ib))].astype(int)), 3)
            r = dict(tau=tau, K=k, kept={ds: int((d.dataset == ds).sum()) for ds in DATASETS},
                     n_train=info["n_train"], auc=auc)
            res.append(r)
            print(json.dumps(r), flush=True)
            json.dump(res, open(f"{ROOT}/match/diag_trim_{os.environ.get('TAUS', 'default')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
