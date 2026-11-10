# SPDX-License-Identifier: Apache-2.0
"""Teacher "bad-video score" s per clip and its distribution on ViPE / Mixkit / OpenVid (user, 2026-10-15;
task spec data_analysis_task.md). Measurement only: no training, no change to training code.

Same clips, latents, eps and teacher as scripts/diag/rho_eval.py (teacher_cond / teacher_cfg rows):
  x_t = forward_process(x, eps, t), eps = CPU randn seeded 1000 * clip_index + T_STUDENT.index(t);
  three teacher x0 predictions at the same x_t (fwd_pred_type="x0", same precision / inference_mode as rho_eval):
    x0_c   : the clip's own caption
    x0_neg : PRESET_CONSTANTS["neg_prompt_wan"] (the training dataloader's neg_condition, VideoLoaderConfig)
    x0_emp : the empty string ""
  d(a, b) = mean over elements of (a - b)^2 (= D_t of rho_eval).
Per clip: D_c, D_neg, D_emp, g_neg = d(x0_c, x0_neg), g_emp = d(x0_c, x0_emp), D_cfg_rec = d(x, x0_c + 4 (x0_c -
x0_neg)) (= the DFD CFG target, w = 5; compared with the stored Dcfg of rho_eval), caption chars / tokens, latent
E x^2, latent frame-difference energy, per-clip element mean of x0_neg / x0_emp.
Per (dataset, t): Gram matrices <u_i, u_j> / d (float64) of x0_c, x0_neg, x0_emp (-> spread of each branch around its
mean, with an exact clip bootstrap) and their element means (cross-dataset comparison of the mean predictions).

GPU (one process per GPU; a unit = dataset:t1,t2 (caption encoded once per clip, on CPU as in rho_eval);
     each t is skipped if <out>/<ds>/t<t>/DONE exists):
  torchrun --nproc_per_node=1 scripts/diag/s_score.py --units vipe:0.999,0.937 openvid:0.999 --out DIR \
      --config fastgen/configs/experiments/WanT2V/config_dmd2.py - trainer.fsdp=True trainer.ddp=False
Report (numpy / scipy / av / PIL, CPU): python scripts/diag/s_score.py --report --out DIR
Plots  (numpy / matplotlib, CPU):        python scripts/diag/s_score.py --plots --out DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

DATASETS = ["vipe", "mixkit", "openvid"]  # = scripts.diag.rho_eval (not imported here: --plots runs without av/torch)
LAT = {"vipe": "runs/delta_diag/n256/vipe/latents_fp16.pt", "mixkit": "runs/delta_diag/n256/mixkit/latents_fp16.pt",
       "openvid": "runs/rho_eval/n256/openvid/latents_fp16.pt"}
# stored teacher D (Dcond / Dcfg) of the same clips, eps and t: check (a) and the neg-branch check
OLD_TEACHER = {"vipe": "runs/openvid_eval/n256/vipe/D_teacher.jsonl",
               "mixkit": "runs/openvid_eval/n256/mixkit/D_teacher.jsonl",
               "openvid": "runs/rho_eval/n256/openvid/D_teacher.jsonl"}
W_CFG = 5.0
BRANCHES = ["c", "neg", "emp"]


def sources(ds, n):
    from scripts.diag.delta_scale import clip_source
    from scripts.diag.teacher_openvid import openvid_source

    return list(openvid_source(n) if ds == "openvid" else clip_source(ds, n))


def run(args):
    import torch

    import fastgen.utils.logging_utils as logger
    from fastgen.datasets.wds_dataloaders import PRESET_CONSTANTS
    from fastgen.utils import basic_utils
    from fastgen.utils.basic_utils import prompt_clean
    from fastgen.utils.distributed import clean_up
    from fastgen.utils.scripts import setup
    from scripts.diag.delta_scale import T_STUDENT
    from scripts.inference.inference_utils import cleanup_unused_modules, init_model, setup_inference_modules

    torch.backends.cuda.matmul.allow_tf32 = False
    config = setup(args, evaluation=True)

    # teacher exactly as rho_eval.build(None, teacher=True)
    model = init_model(config)
    cleanup_unused_modules(model, True)
    net, _, _ = setup_inference_modules(model, config, True, False, model.precision)
    assert model.config.guidance_scale == W_CFG and model.config.skip_layers is None
    ns = model.net.noise_scheduler
    tok = model.net.text_encoder.tokenizer

    def encode_text(caps):
        ctx = {"dtype": model.precision, "device": model.device}
        with basic_utils.inference_mode(model.net.text_encoder, precision_amp=model.precision_amp_enc,
                                        device_type="cuda"):
            return basic_utils.to(model.net.text_encoder.encode(caps), **ctx)

    NEG = PRESET_CONSTANTS["neg_prompt_wan"]
    neg, emp = encode_text([NEG]), encode_text([""])
    logger.info(f"[s_score] neg prompt ({len(NEG)} chars): {NEG!r}")

    def d(a, b):
        return float((a - b).pow(2).mean())

    for unit in args.units:  # "ds:t1,t2": the caption is encoded once per clip (CPU umT5) and reused for every t
        ds, ts = unit.split(":")
        ts = [float(t) for t in ts.split(",")]
        ods = {t: os.path.join(args.out, ds, f"t{t}") for t in ts}
        ts = [t for t in ts if not os.path.exists(os.path.join(ods[t], "DONE"))]
        if not ts:
            logger.info(f"[s_score] {unit} done; skipping")
            continue
        for t in ts:
            os.makedirs(ods[t], exist_ok=True)
        saved = torch.load(LAT[ds])
        src = sources(ds, min(args.n, len(saved["ids"])))
        assert [s[0] for s in src] == saved["ids"][: len(src)], f"{ds}: clip order differs from the stored latents"
        N = len(src)
        old = {r["id"]: r for r in map(json.loads, open(OLD_TEACHER[ds]))}
        keep = {(t, b): [] for t in ts for b in BRANCHES}  # x0 per clip in the net's output dtype (bf16), CPU
        rows = {t: [] for t in ts}
        for i, (cid, _, cap) in enumerate(src):
            x = saved["lat"][i].to("cuda").float().unsqueeze(0)
            cond = encode_text([cap])
            xs = x[0]
            ntok = len(tok(prompt_clean(cap), add_special_tokens=True)["input_ids"])
            for t in ts:
                k = T_STUDENT.index(t)
                eps = torch.randn(x.shape, generator=torch.Generator().manual_seed(1000 * i + k)).to("cuda")
                tt = torch.full((1,), t, device="cuda", dtype=ns.t_precision)
                x_t = ns.forward_process(x, eps, tt).to(model.precision)
                x0 = {}
                with basic_utils.inference_mode(net, precision_amp=model.precision_amp_infer, device_type="cuda"):
                    for b, c in [("c", cond), ("neg", neg), ("emp", emp)]:
                        x0[b] = net(x_t, tt, condition=c, fwd_pred_type="x0")
                    # DFD target as in DMD2Model._apply_classifier_free_guidance (same dtype path as rho_eval)
                    x0_cfg = x0["c"] + (W_CFG - 1) * (x0["c"] - x0["neg"])
                for b in BRANCHES:
                    keep[(t, b)].append(x0[b][0].flatten().cpu())
                x0 = {b: v.float() for b, v in x0.items()}
                r = dict(ds=ds, t=t, idx=i, vid=cid, caption=cap,
                         D_c=d(x, x0["c"]), D_neg=d(x, x0["neg"]), D_emp=d(x, x0["emp"]),
                         g_neg=d(x0["c"], x0["neg"]), g_emp=d(x0["c"], x0["emp"]),
                         D_cfg_rec=d(x, x0_cfg.float()),
                         caption_chars=len(cap), caption_tokens=ntok, caption_truncated=int(ntok > 512),
                         latent_Ex2=float(xs.pow(2).mean()),
                         frame_diff=float((xs[:, 1:] - xs[:, :-1]).pow(2).mean()),  # [C, T, H, W]: latent frames
                         x0neg_mean=float(x0["neg"].mean()), x0emp_mean=float(x0["emp"].mean()),
                         x0neg_Ex2=float(x0["neg"].pow(2).mean()), x0emp_Ex2=float(x0["emp"].pow(2).mean()))
                o = old.get(cid)
                r["old_Dcond"] = o[f"Dcond_{t}"] if o else None
                r["old_Dcfg"] = o[f"Dcfg_{t}"] if o else None
                rows[t].append(r)
                if i % 16 == 0:
                    logger.info(f"[{ds} t={t} {i + 1}/{N}] " + json.dumps(
                        {kk: round(r[kk], 4) for kk in ["D_c", "D_neg", "D_emp", "g_neg", "g_emp", "D_cfg_rec",
                                                        "old_Dcond", "old_Dcfg"] if r[kk] is not None}))
        for t in ts:
            gram, means = {}, {}
            for b in BRANCHES:
                X = torch.stack(keep.pop((t, b))).double()  # [N, d], CPU float64
                gram[b] = (X @ X.T / X.shape[1]).numpy()
                means[b] = X.mean(0).float().view_as(saved["lat"][0])
                del X
            np.savez(os.path.join(ods[t], "gram.npz"), **gram)
            torch.save(means, os.path.join(ods[t], "means.pt"))
            with open(os.path.join(ods[t], "rows.jsonl"), "w") as f:
                for r in rows[t]:
                    f.write(json.dumps(r) + "\n")
            open(os.path.join(ods[t], "DONE"), "w").write(f"{N}\n")
            logger.info(f"[s_score] {ds} t={t} written ({N} clips)")
        del keep, saved
    clean_up()


# ----------------------------------------------------------------------------- report (CPU)


def load(out, t):
    from scripts.diag.rho_eval import PAIR

    data = {}
    for ds in DATASETS:
        p = os.path.join(out, ds, f"t{t}", "rows.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p)]
        n = len(rows)
        pair = np.load(PAIR[ds])[:n, :n]
        G = dict(np.load(os.path.join(out, ds, f"t{t}", "gram.npz")))
        data[ds] = dict(rows=rows, n=n, pair=pair, G=G)
    return data


def shuf(pair, idx):
    p = pair[np.ix_(idx, idx)]
    return p[idx[:, None] != idx[None, :]].mean()


def spread(G, idx):
    """mean over clips of d(u_i, mean_j u_j), from the Gram matrix <u_i, u_j> / d (exact)."""
    g = G[np.ix_(idx, idx)]
    return float(np.mean(np.diag(g)) - g.mean())


def report(args):
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    B = 1000
    out = args.out
    lines = ["# Teacher bad-video score s on ViPE / Mixkit / OpenVid", ""]
    res = {}
    # earlier CFG/cond ratios at t=0.999 (runs/rho_eval/n256/report.md) for check (c)
    old_ratio = {"vipe": 4.22, "mixkit": 5.86, "openvid": 3.25}
    old_rhoc = {"vipe": 0.410, "mixkit": 0.359, "openvid": 0.433}
    csv_rows = []
    for t in [0.999, 0.937]:
        data = load(out, t)
        if not data:
            continue
        lines += [f"## t = {t}", ""]
        for ds, D in data.items():
            n = D["n"]
            D["Dshuf"] = shuf(D["pair"], np.arange(n))
            D["boot"] = [rng.integers(0, n, n) for _ in range(B)]
            D["bshuf"] = np.array([shuf(D["pair"], b) for b in D["boot"]])
            for r in D["rows"]:
                r["s_neg"] = r["D_neg"] - r["D_c"]
                r["s_emp"] = r["D_emp"] - r["D_c"]
                r["s_neg_norm"] = r["s_neg"] / D["Dshuf"]
                r["s_emp_norm"] = r["s_emp"] / D["Dshuf"]
                r["D_shuf"] = D["Dshuf"]
            if t == 0.999:
                csv_rows += D["rows"]

        # ---- checks
        lines += ["### Checks", "",
                  "(a) rho_c from this run vs the earlier teacher_cond rho (same clips / eps / t); per-clip max relative "
                  "difference of D_c vs the stored Dcond and of the reconstructed CFG target D vs the stored Dcfg "
                  "(the stored Dcfg used neg_prompt_wan, so this also checks the neg branch).", "",
                  "| dataset | n | D_shuf | rho_c (this run) | rho_c (earlier report) | max rel |D_c/Dcond-1| | "
                  "max rel |D_cfg/Dcfg-1| |", "|---|---|---|---|---|---|---|"]
        for ds, D in data.items():
            v = lambda k: np.array([r[k] for r in D["rows"]], dtype=float)
            rel_c = np.nanmax(np.abs(v("D_c") / v("old_Dcond") - 1))
            rel_g = np.nanmax(np.abs(v("D_cfg_rec") / v("old_Dcfg") - 1))
            rc = v("D_c").mean() / D["Dshuf"]
            res[f"check_a_{ds}_{t}"] = dict(rho_c=rc, max_rel_Dc=rel_c, max_rel_Dcfg=rel_g, Dshuf=D["Dshuf"])
            lines.append(f"| {ds} | {D['n']} | {D['Dshuf']:.4f} | {rc:.4f} | "
                         f"{old_rhoc[ds] if t == 0.999 else '–'} | {rel_c:.2e} | {rel_g:.2e} |")

        # ---- dataset-level table with clip bootstrap (D_shuf resampled with the clips)
        def stat_fns(D):
            def f(idx, Ds):
                m = {k: np.mean([D["rows"][i][k] for i in idx]) for k in ["D_c", "D_neg", "D_emp", "g_neg", "g_emp",
                                                                          "D_cfg_rec"]}
                rc, rn, re = m["D_c"] / Ds, m["D_neg"] / Ds, m["D_emp"] / Ds
                return {"rho_c": rc, "rho_neg": rn, "rho_emp": re,
                        "shift_neg": rn - 0.5, "shift_emp": re - 0.5, "caption": 0.5 - rc,
                        "g_neg/Dshuf": m["g_neg"] / Ds, "g_emp/Dshuf": m["g_emp"] / Ds,
                        "(c) 1+16(rho_neg/rho_c-1)": 1 + 16 * (rn / rc - 1),
                        "(c') CFG/cond exact": m["D_cfg_rec"] / m["D_c"],
                        "(b) spread x0_neg/Dshuf": spread(D["G"]["neg"], idx) / Ds,
                        "(b) spread x0_emp/Dshuf": spread(D["G"]["emp"], idx) / Ds,
                        "spread x0_c/Dshuf": spread(D["G"]["c"], idx) / Ds}
            return f

        table = {}
        for ds, D in data.items():
            f = stat_fns(D)
            pt = f(np.arange(D["n"]), D["Dshuf"])
            bs = [f(b, s) for b, s in zip(D["boot"], D["bshuf"])]
            table[ds] = {k: (v, *np.nanpercentile([b[k] for b in bs], [2.5, 97.5])) for k, v in pt.items()}
            res[f"summary_{ds}_{t}"] = table[ds]
        keys = list(next(iter(table.values())))
        lines += ["", "### Dataset-level summary (95 % CI: 1000 clip bootstraps, D_shuf resampled with the clips)", "",
                  "| quantity | " + " | ".join(table) + " |", "|---|" + "---|" * len(table)]
        for k in keys:
            lines.append(f"| {k} | " + " | ".join(f"{table[ds][k][0]:.4f} [{table[ds][k][1]:.4f}, "
                                                  f"{table[ds][k][2]:.4f}]" for ds in table) + " |")
        if t == 0.999:
            lines += ["", "Check (c): earlier measured CFG/cond ratio " +
                      ", ".join(f"{ds} {old_ratio[ds]}" for ds in table) + "; gap of 1+16(rho_neg/rho_c-1): " +
                      ", ".join(f"{ds} {table[ds]['(c) 1+16(rho_neg/rho_c-1)'][0] - old_ratio[ds]:+.2f}"
                                for ds in table) +
                      ". The formula assumes <x - x0_c, x0_c - x0_neg> = 0; the exact identity is "
                      "D_cfg = 5 D_c - 4 D_neg + 20 g_neg (row (c'), = D_cfg_rec / D_c)."]
            # mean predictions across datasets (the same eps index i is used in every dataset)
            import torch
            mean = {ds: torch.load(os.path.join(out, ds, f"t{t}", "means.pt")) for ds in data}
            lines += ["", "Distance between the dataset means of each branch, d(m_A, m_B) / mean D_shuf:", "",
                      "| branch | " + " | ".join(f"{a}-{b}" for a, b in [("vipe", "mixkit"), ("vipe", "openvid"),
                                                                        ("mixkit", "openvid")]) + " |", "|---|---|---|---|"]
            for b in BRANCHES:
                cells = []
                for a, c in [("vipe", "mixkit"), ("vipe", "openvid"), ("mixkit", "openvid")]:
                    if a in mean and c in mean:
                        ds_ = (data[a]["Dshuf"] + data[c]["Dshuf"]) / 2
                        cells.append(f"{float((mean[a][b] - mean[c][b]).pow(2).mean()) / ds_:.4f}")
                    else:
                        cells.append("–")
                lines.append(f"| x0_{b} | " + " | ".join(cells) + " |")
            # dataset ratios of the shift term
            if "vipe" in data and "mixkit" in data:
                lines += ["", "Shift terms Mixkit vs ViPE / OpenVid (independent bootstraps):", ""]
                for k in ["shift_neg", "shift_emp", "caption"]:
                    for a, b in [("mixkit", "vipe"), ("mixkit", "openvid"), ("openvid", "vipe")]:
                        if a not in data or b not in data:
                            continue
                        fa, fb = stat_fns(data[a]), stat_fns(data[b])
                        da = np.array([fa(i, s)[k] for i, s in zip(data[a]["boot"], data[a]["bshuf"])])
                        db = np.array([fb(i, s)[k] for i, s in zip(data[b]["boot"], data[b]["bshuf"])])
                        diff = table[a][k][0] - table[b][k][0]
                        lo, hi = np.nanpercentile(da - db, [2.5, 97.5])
                        res[f"diff_{k}_{a}-{b}"] = (diff, lo, hi)
                        lines.append(f"- {k}: {a} - {b} = {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]")

        if t != 0.999:
            continue
        # ---- distributions of s~
        qs = [5, 25, 50, 75, 95]
        lines += ["", "### Quantiles of the normalised score s~ = s / D_shuf", "",
                  "| score | dataset | mean | " + " | ".join(f"{q} %" for q in qs) + " |",
                  "|---|---|---|" + "---|" * len(qs)]
        for k in ["s_neg_norm", "s_emp_norm"]:
            for ds, D in data.items():
                v = np.array([r[k] for r in D["rows"]])
                lines.append(f"| {k} | {ds} | {v.mean():.4f} | " +
                             " | ".join(f"{x:.4f}" for x in np.percentile(v, qs)) + " |")
        lines += ["", "### Right tail of ViPE and OpenVid relative to Mixkit (fraction of clips above a Mixkit quantile; "
                  "95 % CI: both datasets bootstrapped, s~ re-normalised with the resampled D_shuf)", ""]
        if "mixkit" in data:
            for score in ["s_neg_norm", "s_emp_norm"]:
                sm = np.array([r[score] for r in data["mixkit"]["rows"]])
                for ds in [d_ for d_ in ["vipe", "openvid"] if d_ in data]:
                    sv = np.array([r[score] for r in data[ds]["rows"]])
                    for q in [50, 25]:
                        thr = np.percentile(sm, q)
                        frac = (sv > thr).mean()
                        bf = []
                        for bv, bm, sv_s, sm_s in zip(data[ds]["boot"], data["mixkit"]["boot"],
                                                      data[ds]["bshuf"], data["mixkit"]["bshuf"]):
                            sv_b = sv[bv] * data[ds]["Dshuf"] / sv_s
                            sm_b = sm[bm] * data["mixkit"]["Dshuf"] / sm_s
                            bf.append((sv_b > np.percentile(sm_b, q)).mean())
                        lo, hi = np.nanpercentile(bf, [2.5, 97.5])
                        res[f"{ds}_above_mixkit_{score}_q{q}"] = (frac, lo, hi, thr)
                        lines.append(f"- {score}: {ds} clips > Mixkit {q} % quantile ({thr:.4f}): {frac * 100:.1f} % "
                                     f"[{lo * 100:.1f}, {hi * 100:.1f}] ({int((sv > thr).sum())}/{len(sv)})")

        # ---- within-dataset Spearman correlations
        feats = ["caption_chars", "caption_tokens", "latent_Ex2", "frame_diff", "D_c"]
        lines += ["", "### Spearman correlation of s_neg with per-clip features (within dataset; p in brackets)", "",
                  "| dataset | " + " | ".join(feats) + " | s_emp |", "|---|" + "---|" * (len(feats) + 1)]
        for ds, D in data.items():
            s = [r["s_neg"] for r in D["rows"]]
            cells = []
            for fname in feats + ["s_emp"]:
                rr, p = spearmanr(s, [r[fname] for r in D["rows"]])
                res[f"spearman_{ds}_{fname}"] = (rr, p)
                cells.append(f"{rr:+.3f} ({p:.1g})")
            lines.append(f"| {ds} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Truncated captions (> 512 tokens): " + ", ".join(
            f"{ds} {sum(r['caption_truncated'] for r in D['rows'])}" for ds, D in data.items()))

        # ---- qualitative grids
        if not args.no_frames:
            lines += ["", "### Qualitative check", "", "`frames.html` (and `frames_<ds>_<top|bottom>.png`): first / "
                      "middle / last frame of the 12 clips with the highest and the lowest s_neg in ViPE, Mixkit and OpenVid."]
            frames(out, data)

    # ---- per-clip CSV (t = 0.999)
    cols = ["ds", "t", "idx", "vid", "caption", "D_c", "D_neg", "D_emp", "g_neg", "g_emp", "s_neg", "s_emp",
            "s_neg_norm", "s_emp_norm", "D_shuf", "caption_chars", "caption_tokens", "caption_truncated", "latent_Ex2",
            "frame_diff", "x0neg_mean", "x0emp_mean", "x0neg_Ex2", "x0emp_Ex2", "D_cfg_rec", "old_Dcond", "old_Dcfg"]
    with open(os.path.join(out, "s_scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(csv_rows)
    json.dump(res, open(os.path.join(out, "report.json"), "w"), indent=1, default=float)
    open(os.path.join(out, "tables.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def frames(out, data):
    """First / middle / last frame (training transform: resize small side, center crop 832x480, frames 0:81)."""
    import base64
    import html
    import io

    from PIL import Image, ImageDraw

    from scripts.diag.delta_scale import decode

    W, H, SEQ, TW = 832, 480, 81, 208  # thumbnails 208x120
    sel = {}
    for ds in DATASETS:
        if ds not in data:
            continue
        order = sorted(data[ds]["rows"], key=lambda r: -r["s_neg"])
        sel[ds] = {"top": order[:12], "bottom": order[-12:][::-1]}
    want = {r["vid"] for ds in sel for g in sel[ds].values() for r in g}
    thumbs = {}
    for ds in sel:
        for cid, mp4, _ in sources(ds, data[ds]["n"]):
            if cid not in want:
                continue
            v = decode(mp4)[:SEQ]
            ims = []
            for fi in [0, SEQ // 2, SEQ - 1]:
                im = Image.fromarray(v[fi])
                s = max(W / im.width, H / im.height)  # resize small side, aspect preserving
                im = im.resize((max(W, round(im.width * s)), max(H, round(im.height * s))), Image.BILINEAR)
                l, tp = (im.width - W) // 2, (im.height - H) // 2
                ims.append(im.crop((l, tp, l + W, tp + H)).resize((TW, TW * H // W), Image.BILINEAR))
            thumbs[cid] = ims
    page = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>",
            "<title>s_neg extremes</title><style>body{font:13px system-ui,sans-serif;margin:16px;background:#fff;"
            "color:#111}h2{margin:24px 0 8px}.r{display:flex;gap:8px;align-items:center;margin:4px 0;flex-wrap:wrap}"
            ".r img{width:208px;height:120px}.c{max-width:360px}</style></head><body>",
            "<h1>Teacher score s_neg: highest / lowest 12 clips</h1><p>Frames 0 / 40 / 80 after the training crop. "
            "s_neg = D_neg − D_c (t = 0.999); s~ = s / D_shuf.</p>"]
    for ds in sel:
        for g in ["top", "bottom"]:
            rows = sel[ds][g]
            page.append(f"<h2>{ds}: {'highest' if g == 'top' else 'lowest'} s_neg</h2>")
            grid = Image.new("RGB", (3 * TW + 4 * 4 + 520, len(rows) * (TW * H // W + 4) + 4), "white")
            dr = ImageDraw.Draw(grid)
            for j, r in enumerate(rows):
                y = 4 + j * (TW * H // W + 4)
                cells = []
                for q, im in enumerate(thumbs[r["vid"]]):
                    grid.paste(im, (4 + q * (TW + 4), y))
                    b = io.BytesIO()
                    im.save(b, format="JPEG", quality=85)
                    cells.append(f"<img src='data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}'>")
                cap = r["caption"][:80]
                txt = f"s_neg {r['s_neg']:.3f}  s~ {r['s_neg_norm']:.3f}  D_c {r['D_c']:.3f}"
                dr.text((3 * (TW + 4) + 8, y + 30), txt, fill="black")
                dr.text((3 * (TW + 4) + 8, y + 50), cap, fill="black")
                dr.text((3 * (TW + 4) + 8, y + 70), r["vid"], fill="gray")
                page.append(f"<div class='r'>{''.join(cells)}<div class='c'><b>{html.escape(txt)}</b><br>"
                            f"{html.escape(cap)}<br><small>{html.escape(r['vid'])}</small></div></div>")
            grid.save(os.path.join(out, f"frames_{ds}_{g}.png"))
    page.append("</body></html>")
    open(os.path.join(out, "frames.html"), "w").write("\n".join(page))


def plots(args):
    """Overlaid histograms of s~_neg and s~_emp (t = 0.999) from s_scores.csv; numpy + matplotlib only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(os.path.join(args.out, "s_scores.csv"))))
    # categorical slots 1-3 of the dataviz reference palette (validated all-pairs, light surface)
    color = {"vipe": "#2a78d6", "mixkit": "#eb6834", "openvid": "#1baf7a"}
    label = {"vipe": "ViPE (synthetic)", "mixkit": "Mixkit (real)", "openvid": "OpenVidHD (real)"}
    for k, name in [("s_neg_norm", "s~_neg = (D_neg − D_c) / D_shuf"), ("s_emp_norm", "s~_emp = (D_emp − D_c) / D_shuf")]:
        v = {ds: np.array([float(r[k]) for r in rows if r["ds"] == ds]) for ds in DATASETS}
        v = {ds: a for ds, a in v.items() if len(a)}
        allv = np.concatenate(list(v.values()))
        bins = np.linspace(np.percentile(allv, 0.5), np.percentile(allv, 99.5), 41)
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
        fig.patch.set_facecolor("#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        top = 0
        for ds, a in v.items():
            h, _, _ = ax.hist(np.clip(a, bins[0], bins[-1]), bins=bins, density=True, histtype="stepfilled",
                              color=color[ds], alpha=0.18, linewidth=0)
            ax.hist(np.clip(a, bins[0], bins[-1]), bins=bins, density=True, histtype="step", color=color[ds],
                    linewidth=2, label=f"{label[ds]}  n={len(a)}, median {np.median(a):.3f}")
            top = max(top, h.max())
        for ds, a in v.items():
            ax.axvline(np.median(a), color=color[ds], linewidth=1.5, linestyle=(0, (4, 3)))
        ax.set_ylim(0, top * 1.15)
        ax.set_xlabel(name + "   (t = 0.999; values outside the 0.5-99.5 % range clipped to the edge bins)",
                      color="#52514e", fontsize=8.5)
        ax.set_ylabel("density", color="#52514e")
        ax.set_title(f"Teacher score {k.replace('_norm', '')} per clip, normalised by D_shuf", loc="left",
                     color="#0b0b0b", fontsize=11)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color("#b5b3ad")
        ax.tick_params(colors="#52514e", labelsize=8)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"hist_{k}.png"), facecolor=fig.get_facecolor())
        plt.close(fig)
        print("wrote", os.path.join(args.out, f"hist_{k}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="*", default=[], help="dataset:t[,t...], e.g. vipe:0.999,0.937")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no_frames", action="store_true")
    ap.add_argument("--plots", action="store_true")
    if "--report" in sys.argv:
        report(ap.parse_args())
        return
    if "--plots" in sys.argv:
        plots(ap.parse_args())
        return
    from fastgen.utils.scripts import parse_args

    run(parse_args(ap))


if __name__ == "__main__":
    main()
