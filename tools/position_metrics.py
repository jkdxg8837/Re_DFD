# SPDX-License-Identifier: Apache-2.0
"""Position metrics between real and synthetic (task2.md 0.4 / 0.5 / 0.6). CPU only; run with .venv-vbench
(numpy, scipy, scikit-learn, matplotlib, torch).

Inputs: the cached sets of tools/phaseab_gpu.py under runs/phaseAB/latents/<set>/ (latents.pt fp16 [N, 16, 21, 60,
104], stats.npy, feat_a.npy, feat_b.npy, meta.jsonl). References: R_mx, R_ov (real), S_vipe (synthetic).

Metrics (point estimate + 95 % CI from 1000 bootstrap resamples of the evaluated set's samples):
  pi       <x - mu_S, mu_R - mu_S> / ||mu_R - mu_S||^2 on full flattened latents; mu_R from R_mx u R_ov (pi),
           R_mx only (pi_mx), R_ov only (pi_ov)
  probe    logit of the frozen linear probe (real = 1), feature variants a (pooled latent stats) and b (teacher feats)
  lowfreq  mean luminance, HSV saturation, dark-pixel ratio (decoded pixels), per-channel latent means
  FD / kNN Frechet distance and k-NN (k = 5) precision / recall vs each reference set on feature b, computed in the
           128-d PCA space of the standardised feature b of the three references (fitted once, saved with the probe)

Subcommands:
  mu                          mu_R, mu_S, unit direction -> runs/phaseAB/mu_dir.pt
  fit                         probes: 5-fold CV (logistic regression, C by inner CV; 2-layer MLP), Spearman(logit, pi)
                              on held-out folds; final probes on all references -> runs/phaseAB/probe/ (frozen)
  apply --set S [--ref_suffix _blur<sigma>]   metrics of one set -> runs/phaseAB/metrics/<S>[<suffix>].json
  blurfit --gen A1_w5         sigma whose blurred-R_mx spectrum best matches the generator's -> runs/phaseAB/blur/
  leak --sigma S              linear probe CV on the blurred references (leakage check)
  a3                          A3 report (needs runs/phaseAB/A3/<ds>/a3.pt) -> runs/phaseAB/A3/a3.json + plot
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

ROOT = os.environ.get("PHASEAB_ROOT", "runs/phaseAB")  # data_prep.md A-1..A-3 redo: runs/phaseAB/v2
REFS = ["R_mx", "R_ov", "S_vipe"]
NBOOT = 1000


def ld(name, what):
    d = os.path.join(ROOT, "latents", name)
    if what == "lat":
        import torch

        return torch.load(os.path.join(d, "latents.pt")).float().flatten(1).numpy()
    if what == "meta":
        return [json.loads(l) for l in open(os.path.join(d, "meta.jsonl"))]
    return np.load(os.path.join(d, f"{what}.npy")).astype(np.float64 if what == "stats" else np.float32)


def boot_mean(v, rng, n=NBOOT):
    v = np.asarray(v, dtype=np.float64)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    b = v[idx].mean(1)
    return {"mean": float(v.mean()), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "n": len(v)}


def ci_of(stat_fn, n_items, rng, point, n=NBOOT):
    vals = [stat_fn(rng.integers(0, n_items, n_items)) for _ in range(n)]
    return {"mean": float(point), "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            "n": n_items}


# ------------------------------------------------------------------ mu / pi
def cmd_mu(args):
    import torch

    mx, ov, sv = (ld(s, "lat").astype(np.float64) for s in REFS)
    mu_mx, mu_ov, mu_S = mx.mean(0), ov.mean(0), sv.mean(0)
    mu_R = np.concatenate([mx, ov]).mean(0)
    dvec = mu_R - mu_S
    shp = (16, 21, 60, 104)
    T = lambda v: torch.from_numpy(v.astype(np.float32).reshape(shp))
    # format agreed with the Phase B residual logger: latent-shaped float32 tensors; mu_dir unnormalised, u unit
    out = {"mu_R": T(mu_R), "mu_S": T(mu_S), "mu_dir": T(dvec), "u": T(dvec / np.linalg.norm(dvec)),
           "mu_R_mx": T(mu_mx), "mu_R_ov": T(mu_ov), "norm_sq_per_elem": float((dvec ** 2).mean()),
           "note": "[16, 21, 60, 104]; mu_R = mean of R_mx u R_ov, mu_S = mean of S_vipe; mu_dir = mu_R - mu_S, "
                   "u = mu_dir / ||mu_dir||"}
    torch.save(out, os.path.join(ROOT, "mu_dir.pt"))
    cos = float(np.dot(mu_mx - mu_S, mu_ov - mu_S) / np.linalg.norm(mu_mx - mu_S) / np.linalg.norm(mu_ov - mu_S))
    print(f"||mu_R - mu_S||^2 / d = {out['norm_sq_per_elem']:.6f}; cos(mu_mx - mu_S, mu_ov - mu_S) = {cos:.3f}")


def pis(X, mu):
    res = {}
    for key, mr in [("pi", mu["mu_R"]), ("pi_mx", mu["mu_R_mx"]), ("pi_ov", mu["mu_R_ov"])]:
        dv = mr - mu["mu_S"]
        res[key] = (X - mu["mu_S"]) @ dv / float(dv @ dv)
    return res


def load_mu():
    import torch

    m = torch.load(os.path.join(ROOT, "mu_dir.pt"))
    return {k: m[k].flatten().numpy().astype(np.float64) for k in ["mu_R", "mu_S", "mu_R_mx", "mu_R_ov"]}


# ------------------------------------------------------------------ probes
def make_lr():
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-5, 1, 7), cv=5, scoring="roc_auc",
                                                                class_weight="balanced", max_iter=5000,
                                                                random_state=0))


def make_mlp():
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, max_iter=1000,
                                                         early_stopping=True, validation_fraction=0.15,
                                                         random_state=0))


def balance(X, y):
    """Duplicate synthetic samples so both classes have equal weight (MLPClassifier has no class_weight)."""
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    rep = np.resize(i0, len(i1)) if len(i0) < len(i1) else i0
    idx = np.concatenate([i1, rep])
    return X[idx], y[idx]


def ref_xy(variant, suffix=""):
    X = np.concatenate([ld(s + suffix, f"feat_{variant}") for s in REFS])
    y = np.concatenate([np.ones(300), np.ones(300), np.zeros(300)]).astype(int)
    return X, y


def cv_probe(X, y, pi=None, mlp=True):
    from scipy.stats import spearmanr
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    folds = []
    held_logit = np.zeros(len(y))
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)):
        lr = make_lr().fit(X[tr], y[tr])
        lg = lr.decision_function(X[te])
        held_logit[te] = lg
        f = {"fold": k, "lr_acc": float(accuracy_score(y[te], lg > 0)), "lr_auc": float(roc_auc_score(y[te], lg)),
             "lr_C": float(lr[-1].C_[0])}
        if mlp:
            m = make_mlp().fit(*balance(X[tr], y[tr]))
            pm = m.predict_proba(X[te])[:, 1]
            f.update(mlp_acc=float(accuracy_score(y[te], pm > 0.5)), mlp_auc=float(roc_auc_score(y[te], pm)))
        folds.append(f)
    out = {"folds": folds}
    for k in folds[0]:
        if k != "fold":
            out[f"mean_{k}"] = float(np.mean([f[k] for f in folds]))
    if pi is not None:
        out["spearman_heldout_logit_vs_pi"] = float(spearmanr(held_logit, pi).statistic)
        for c, name in [(1, "real"), (0, "synthetic")]:
            out[f"spearman_within_{name}"] = float(spearmanr(held_logit[y == c], pi[y == c]).statistic)
    return out


def cmd_fit(args):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    mu = load_mu()
    pi = np.concatenate([pis(ld(s, "lat"), mu)["pi"] for s in REFS])
    os.makedirs(os.path.join(ROOT, "probe"), exist_ok=True)
    res = {}
    for v in ["a", "b"]:
        X, y = ref_xy(v)
        res[v] = cv_probe(X, y, pi)
        final = make_lr().fit(X, y)
        with open(os.path.join(ROOT, "probe", f"linear_{v}.pkl"), "wb") as f:
            pickle.dump(final, f)
        res[v]["final_C"] = float(final[-1].C_[0])
        print(v, json.dumps({k: x for k, x in res[v].items() if k != "folds"}))
    Xb, _ = ref_xy("b")
    sc = StandardScaler().fit(Xb)
    pca = PCA(128, random_state=0).fit(sc.transform(Xb))
    with open(os.path.join(ROOT, "probe", "fd_space_b.pkl"), "wb") as f:
        pickle.dump({"scaler": sc, "pca": pca}, f)
    res["fd_space"] = {"pca_dims": 128, "explained_variance": float(pca.explained_variance_ratio_.sum())}
    with open(os.path.join(ROOT, "probe", "probe_cv.json"), "w") as f:
        json.dump(res, f, indent=1)


# ------------------------------------------------------------------ distances
def frechet(a, b):
    from scipy import linalg

    m1, m2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs, _ = linalg.sqrtm(s1 @ s2, disp=False)
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs.real))


def knn_pr(real, fake, k=5):
    """Kynkaenniemi et al. 2019: precision = share of fake inside some real k-NN ball, recall = reverse."""
    from scipy.spatial.distance import cdist

    def radii(x):
        d = cdist(x, x)
        return np.sort(d, axis=1)[:, k]  # index 0 is the point itself

    rr, rf = radii(real), radii(fake)
    d = cdist(fake, real)
    prec = float((d <= rr[None, :]).any(1).mean())
    rec = float((d.T <= rf[None, :]).any(1).mean())
    return prec, rec


def cmd_apply(args):
    rng = np.random.default_rng(0)
    name, suf = args.set, args.ref_suffix or ""
    mu = load_mu()
    out = {"set": name, "ref_suffix": suf}
    meta = ld(name, "meta") if os.path.exists(os.path.join(ROOT, "latents", name, "meta.jsonl")) else []
    src = np.array([m.get("source", "ref") for m in meta])
    if os.path.exists(os.path.join(ROOT, "latents", name, "latents.pt")):
        P = pis(ld(name, "lat"), mu)
        for k, v in P.items():
            out[k] = boot_mean(v, rng)
    else:
        P = None
    st = ld(name, "stats") if os.path.exists(os.path.join(ROOT, "latents", name, "stats.npy")) else None
    if st is not None:
        for j, k in enumerate(["luminance", "saturation", "dark_ratio"]):
            out[k] = boot_mean(st[:, j], rng)
        out["latent_channel_means"] = st[:, 3:].mean(0).tolist()
    logits = {}
    for v in ["a", "b"]:
        with open(os.path.join(ROOT, "probe", f"linear_{v}.pkl"), "rb") as f:
            probe = pickle.load(f)
        logits[v] = probe.decision_function(ld(name, f"feat_{v}"))
        out[f"probe_logit_{v}"] = boot_mean(logits[v], rng)
        out[f"probe_real_frac_{v}"] = boot_mean((logits[v] > 0).astype(float), rng)
        if suf:  # task2.md A1: baseline = the frozen probe on the low-passed references
            for r in REFS:
                out[f"ref{suf}_probe_logit_{v}_{r}"] = boot_mean(probe.decision_function(ld(r + suf, f"feat_{v}")), rng)
    with open(os.path.join(ROOT, "probe", "fd_space_b.pkl"), "rb") as f:
        sp = pickle.load(f)
    proj = lambda F_: sp["pca"].transform(sp["scaler"].transform(F_))
    G = proj(ld(name, "feat_b"))
    for r in REFS:
        R = proj(ld(r + suf, "feat_b"))
        fd = frechet(G, R)
        out[f"FD_{r}"] = ci_of(lambda i: frechet(G[i], R), len(G), rng, fd, n=200)
        p, rc = knn_pr(R, G)
        bs = [knn_pr(R, G[rng.integers(0, len(G), len(G))]) for _ in range(200)]
        out[f"knn_precision_{r}"] = {"mean": p, "ci": np.percentile([b[0] for b in bs], [2.5, 97.5]).tolist()}
        out[f"knn_recall_{r}"] = {"mean": rc, "ci": np.percentile([b[1] for b in bs], [2.5, 97.5]).tolist()}
    out["bootstrap_note"] = "pi / probe / lowfreq: 1000 resamples; FD / kNN: 200 resamples of the evaluated set"
    # split by prompt source (generated sets)
    if len(set(src)) > 1:
        out["by_source"] = {}
        for s_ in sorted(set(src)):
            m = src == s_
            d = {}
            if P is not None:
                d["pi"] = boot_mean(P["pi"][m], rng)
            if st is not None:
                d["luminance"] = boot_mean(st[m, 0], rng)
                d["saturation"] = boot_mean(st[m, 1], rng)
            for v in ["a", "b"]:
                d[f"probe_logit_{v}"] = boot_mean(logits[v][m], rng)
            for r in REFS:
                R = proj(ld(r + suf, "feat_b"))
                d[f"FD_{r}"] = frechet(G[m], R)
            out["by_source"][s_] = d
    os.makedirs(os.path.join(ROOT, "metrics"), exist_ok=True)
    with open(os.path.join(ROOT, "metrics", f"{name}{suf}.json"), "w") as f:
        json.dump(out, f, indent=1)
    brief = {k: (v["mean"] if isinstance(v, dict) and "mean" in v else None) for k, v in out.items()}
    print(name + suf, json.dumps({k: round(v, 4) for k, v in brief.items() if v is not None}))


# ------------------------------------------------------------------ blur control
def cmd_blurfit(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from tools.phaseab_util import gaussian_blur_hw, radial_power_spectrum

    gen = torch.load(os.path.join(ROOT, "latents", args.gen, "latents.pt")).float()
    ref = torch.load(os.path.join(ROOT, "latents", "R_mx", "latents.pt")).float()
    sg = radial_power_spectrum(gen)
    grid = np.round(np.arange(0.0, 4.01, 0.1), 2)
    errs, spec = [], {}
    for s in grid:
        sr = torch.zeros_like(sg)
        for i0 in range(0, len(ref), 25):
            sr += radial_power_spectrum(gaussian_blur_hw(ref[i0 : i0 + 25], float(s))) * len(ref[i0 : i0 + 25])
        sr /= len(ref)
        spec[float(s)] = sr
        errs.append(float(((sr.log() - sg.log()) ** 2)[1:].mean()))  # skip DC
    best = float(grid[int(np.argmin(errs))])
    od = os.path.join(ROOT, "blur")
    os.makedirs(od, exist_ok=True)
    f = np.linspace(0, 0.5, len(sg))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(f, sg.numpy(), "k-", lw=2, label=args.gen)
    for s in [0.0, best, min(4.0, best + 1.0)]:
        ax[0].semilogy(f, spec[float(np.round(s, 2))].numpy(), label=f"R_mx blur sigma={s:g}")
    ax[0].set_xlabel("spatial frequency (cycles / latent pixel)")
    ax[0].set_ylabel("radially averaged power")
    ax[0].legend()
    ax[1].plot(grid, errs, "o-")
    ax[1].axvline(best, color="r", ls="--")
    ax[1].set_xlabel("sigma (latent pixels)")
    ax[1].set_ylabel("mean squared log-spectrum difference")
    fig.tight_layout()
    tag = "" if args.gen == "A1_w5" else f"_{args.gen}"  # A1_w5 = the spec's fit (blurfit.json); others supplementary
    fig.savefig(os.path.join(od, f"spectra{tag}.png"), dpi=120)
    json.dump({"gen": args.gen, "best_sigma": best, "grid": grid.tolist(), "errors": errs,
               "spectrum_gen": sg.tolist(), "spectrum_ref_best": spec[best].tolist()},
              open(os.path.join(od, f"blurfit{tag}.json"), "w"), indent=1)
    print(f"best sigma {best} (err {min(errs):.4f}; sigma 0: {errs[0]:.4f})")


def cmd_leak(args):
    suf = f"_blur{args.sigma:g}"
    res = {}
    for v in ["a", "b"]:
        X, y = ref_xy(v, suf)
        res[v] = cv_probe(X, y, mlp=False)
        print(v, json.dumps({k: x for k, x in res[v].items() if k != "folds"}))
    base = json.load(open(os.path.join(ROOT, "probe", "probe_cv.json")))
    for v in ["a", "b"]:
        res[v]["acc_drop_vs_unblurred"] = base[v]["mean_lr_acc"] - res[v]["mean_lr_acc"]
        res[v]["auc_drop_vs_unblurred"] = base[v]["mean_lr_auc"] - res[v]["mean_lr_auc"]
    json.dump(res, open(os.path.join(ROOT, "probe", f"leak{suf}.json"), "w"), indent=1)


# ------------------------------------------------------------------ A3
def cmd_a3(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    # pair mean-squared-distance matrices of the shift-report clips (= scripts/diag/rho_eval.PAIR, used by s_score)
    PAIR = {"vipe": "runs/delta_diag/n256/vipe/pair_msd.npy", "mixkit": "runs/delta_diag/n256/mixkit/pair_msd.npy",
            "openvid": "runs/openvid_eval/n256/openvid/pair_msd.npy"}

    rng = np.random.default_rng(0)
    mu = torch.load(os.path.join(ROOT, "mu_dir.pt"))
    u = mu["u"].flatten().double().numpy()
    T4 = [0.999, 0.937, 0.833, 0.624]
    tgrid = np.round(np.linspace(0.05, 0.999, 200), 4)
    res, fig = {}, plt.figure(figsize=(6, 4))
    for ds in ["vipe", "mixkit", "openvid"]:
        a = torch.load(os.path.join(ROOT, "A3", ds, "a3.pt"))
        G = a["gram_b"].numpy()
        N, d = G.shape[0], a["d"]
        pm = np.load(PAIR[ds])[:N, :N]  # pair mean squared distance matrix (D_shuf), as s_score
        off = ~np.eye(N, dtype=bool)
        dshuf = float(pm[off].mean())

        def stat(idx):
            gi = G[np.ix_(idx, idx)]
            p = pm[np.ix_(idx, idx)]
            o = idx[:, None] != idx[None, :]
            return gi.mean() / p[o].mean()

        point = float(G.mean() / dshuf)  # ||b_bar||^2 / d / D_shuf (exact from the Gram matrix)
        bbar = a["bbar"].double().numpy()
        lam, bk = a["lam"].double().numpy(), a["bk"].double().numpy()
        snr = lambda t: (1 - t) ** 2 / t ** 2
        r2 = lambda t: float(((bk / (1 + snr(t) * lam)) ** 2).sum() / d)  # top-64 part, per element
        rest = float(a["bbar_sq_total"] - (bk ** 2).sum()) / d  # outside the top-64 span (not attenuated here)
        res[ds] = {"N": N, "D_shuf": dshuf, "rho_c_t0.999": float(a["D_c"].mean() / dshuf),
                   "bbar_sq_over_Dshuf": ci_of(stat, N, rng, point),
                   "cos_bbar_mu_dir": float(bbar @ u / np.linalg.norm(bbar)),
                   "bbar_share_in_top64": float((bk ** 2).sum() / a["bbar_sq_total"]),
                   "lambda_top5": lam[:5].tolist(),
                   "r_sq_per_elem_at_student_t": {str(t): r2(t) for t in T4},
                   "r_sq_rest_outside_top64_per_elem": rest,
                   "r_sq_grid": [r2(t) for t in tgrid]}
        plt.semilogy(tgrid, res[ds]["r_sq_grid"], label=ds)
        plt.scatter(T4, [r2(t) for t in T4], s=12)
    plt.xlabel("t")
    plt.ylabel("predicted ||r(t)||^2 / d (top-64 PCA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "A3", "r_curve.png"), dpi=120)
    res["t_grid"] = tgrid.tolist()
    json.dump(res, open(os.path.join(ROOT, "A3", "a3.json"), "w"), indent=1)
    for ds in ["vipe", "mixkit", "openvid"]:
        r = res[ds]
        print(ds, f"rho_c={r['rho_c_t0.999']:.3f} ||b||^2/Dshuf={r['bbar_sq_over_Dshuf']['mean']:.4f} "
                  f"CI={r['bbar_sq_over_Dshuf']['ci']} cos(b, mu_dir)={r['cos_bbar_mu_dir']:.3f} "
                  f"top64 share={r['bbar_share_in_top64']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mu", "fit", "apply", "blurfit", "leak", "a3"])
    ap.add_argument("--set")
    ap.add_argument("--gen", default="A1_w5")
    ap.add_argument("--sigma", type=float)
    ap.add_argument("--ref_suffix")
    args = ap.parse_args()
    {"mu": cmd_mu, "fit": cmd_fit, "apply": cmd_apply, "blurfit": cmd_blurfit, "leak": cmd_leak,
     "a3": cmd_a3}[args.cmd](args)


if __name__ == "__main__":
    main()
