# SPDX-License-Identifier: Apache-2.0
"""rho_t = D_t / D_shuf on OpenVidHD for teacher and students, plus the trained students on ViPE / Mixkit
(user, 2026-10-12).

Definitions are those of scripts/diag/delta_scale.py and scripts/diag/teacher_openvid.py:
  D_t    = mean over clips of mean (x - x0_hat(x_t))^2, x_t = forward_process(x, eps, t), t in T_STUDENT,
           eps = CPU randn seeded 1000 * clip_index + t_index (identical across datasets), caption = the clip's own;
  D_shuf = mean over pairs i != j of mean (x_i - x_j)^2 within the dataset (pair_msd.npy);
  x0_hat = student one-step x0 prediction, or the teacher's conditional x0 (teacher_cond) and its CFG w=5 x0 with
           neg_prompt_wan (teacher_cfg, = DMD2Model._dfd_guided_teacher_x0, the DFD training target).
Latents: ViPE / Mixkit from runs/delta_diag/n256/<ds>/latents_fp16.pt (the 256 clips of the earlier runs); OpenVid
from data/openvid_prep (same 256 clips / transform as teacher_openvid.py), encoded once and saved as
<out>/openvid/latents_fp16.pt. All new D_t use the fp16-stored latents (the earlier student runs used fp32 latents;
--check_n recomputes E0 on the first clips of ViPE / Mixkit and logs the difference to the stored D_E0).

Run (one process per dataset; each model -> <out>/<ds>/D_<model>.jsonl, skipped if it exists):
  torchrun --nproc_per_node=1 scripts/diag/rho_eval.py --dataset openvid|vipe|mixkit --models teacher E0 ... \
      --n 256 --out DIR --config fastgen/configs/experiments/WanT2V/config_dmd2.py - trainer.fsdp=True trainer.ddp=False
Report (numpy only): python scripts/diag/rho_eval.py --report --out DIR [--n N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from scripts.diag.delta_scale import CKPTS, T_STUDENT, decode

MODELS = {
    "E0": CKPTS["E0"],
    "REF": CKPTS["REF"],
    "E1a_l40s_it40": "FASTGEN_OUTPUT/fastgen/wan_dmd2/E1a_seed1_l40s/checkpoints/0000040",
    "E2_l40s_it40": "FASTGEN_OUTPUT/fastgen/wan_dmd2/E2_seed1_l40s/checkpoints/0000040",
    "E1dmd_l40s_it40": "FASTGEN_OUTPUT/fastgen/wan_dmd2/E1dmd_seed1_l40s/checkpoints/0000040",
}
DATASETS = ["vipe", "mixkit", "openvid"]
# earlier results reused by the report (same clips, eps, t)
OLD = {("vipe", "E0"): "runs/delta_diag/n256/vipe/D_E0.jsonl", ("vipe", "REF"): "runs/delta_diag/n256/vipe/D_REF.jsonl",
       ("mixkit", "E0"): "runs/delta_diag/n256/mixkit/D_E0.jsonl",
       ("mixkit", "REF"): "runs/delta_diag/n256/mixkit/D_REF.jsonl",
       ("vipe", "teacher"): "runs/openvid_eval/n256/vipe/D_teacher.jsonl",
       ("mixkit", "teacher"): "runs/openvid_eval/n256/mixkit/D_teacher.jsonl"}
PAIR = {"vipe": "runs/delta_diag/n256/vipe/pair_msd.npy", "mixkit": "runs/delta_diag/n256/mixkit/pair_msd.npy",
        "openvid": "runs/openvid_eval/n256/openvid/pair_msd.npy"}


def run(args):
    import torch

    import fastgen.utils.logging_utils as logger
    from fastgen.datasets.wds_dataloaders import PRESET_CONSTANTS, transform_video
    from fastgen.utils import basic_utils
    from fastgen.utils.distributed import clean_up
    from fastgen.utils.scripts import setup
    from scripts.diag.teacher_openvid import openvid_source
    from scripts.inference.inference_utils import (cleanup_unused_modules, init_checkpointer, init_model,
                                                   load_checkpoint, setup_inference_modules)

    config = setup(args, evaluation=True)
    od = os.path.join(args.out, args.dataset)
    os.makedirs(od, exist_ok=True)

    def build(ckpt, teacher):
        model = init_model(config)
        if ckpt:
            it, _ = load_checkpoint(init_checkpointer(config), model, ckpt, config)
            assert it is not None, f"checkpoint not loaded: {ckpt}"
        cleanup_unused_modules(model, teacher)
        t_, student, vae = setup_inference_modules(model, config, teacher, not teacher, model.precision)
        return model, (t_ if teacher else student), vae

    def encode_text(model, caps):
        ctx = {"dtype": model.precision, "device": model.device}
        with basic_utils.inference_mode(model.net.text_encoder, precision_amp=model.precision_amp_enc,
                                        device_type="cuda"):
            return basic_utils.to(model.net.text_encoder.encode(caps), **ctx)

    # ---- latents and captions
    lat_path = os.path.join(od if args.dataset == "openvid" else os.path.join("runs/delta_diag/n256", args.dataset),
                            "latents_fp16.pt")
    if args.dataset == "openvid":
        src = list(openvid_source(args.n))
        caps = [c for _, _, c in src]
        if not os.path.exists(lat_path):
            model, _, vae = build(MODELS["E0"], False)  # VAE only; same as teacher_openvid.py --mode openvid
            ctx = {"dtype": model.precision, "device": model.device}
            T = config.model.input_shape[1]
            seq, W, H = (T - 1) * 4 + 1, config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8
            lat = []
            for n, (cid, mp4, _) in enumerate(src):
                px = transform_video(torch.from_numpy(decode(mp4)), seq, (W, H))["real"]
                with basic_utils.inference_mode(vae, precision_amp=model.precision_amp_enc, device_type="cuda"):
                    lat.append(model.net.vae.encode(px.unsqueeze(0).to(**ctx), mode="argmax").float()[0].cpu())
                if n % 32 == 0:
                    logger.info(f"[openvid encode {n + 1}/{len(src)}] {cid}")
            # consistency with the D_shuf of the earlier OpenVid run (pair matrix from fp32 latents)
            X = torch.stack(lat).to("cuda").flatten(1)
            sq = X.pow(2).sum(1)
            pair = ((sq[:, None] + sq[None, :] - 2 * X @ X.T) / X.shape[1]).clamp_min(0).cpu().numpy()
            del X
            old = np.load(PAIR["openvid"])[: len(lat), : len(lat)]
            off = ~np.eye(len(lat), dtype=bool)
            logger.info(f"[openvid encode] D_shuf new {pair[off].mean():.5f} vs stored {old[off].mean():.5f}; "
                        f"max |pair diff| {np.abs(pair - old).max():.2e}")
            torch.save({"ids": [s[0] for s in src], "lat": torch.stack(lat).half()}, lat_path)
            del model, vae
            torch.cuda.empty_cache()
        saved = torch.load(lat_path)
        assert saved["ids"][: len(src)] == [s[0] for s in src], "OpenVid clip order differs"
    else:
        from scripts.diag.delta_scale import clip_source

        saved = torch.load(lat_path)
        src = list(clip_source(args.dataset, min(args.n, len(saved["ids"]))))
        assert [s[0] for s in src] == saved["ids"][: len(src)], "clip order differs from delta_scale run"
        caps = [c for _, _, c in src]
    ids = [s[0] for s in src]
    N = len(ids)

    def d_rows(name, model, net, teacher, n):
        ns = model.net.noise_scheduler
        neg = encode_text(model, [PRESET_CONSTANTS["neg_prompt_wan"]]) if teacher else None
        rows = []
        for i in range(n):
            x = saved["lat"][i].to("cuda").float().unsqueeze(0)
            cond = encode_text(model, [caps[i]])
            r = {"id": ids[i]}
            for k, t in enumerate(T_STUDENT):
                eps = torch.randn(x.shape, generator=torch.Generator().manual_seed(1000 * i + k)).to("cuda")
                tt = torch.full((1,), t, device="cuda", dtype=ns.t_precision)
                x_t = ns.forward_process(x, eps, tt).to(model.precision)
                with basic_utils.inference_mode(net, precision_amp=model.precision_amp_infer, device_type="cuda"):
                    x0 = net(x_t, tt, condition=cond, fwd_pred_type="x0")
                    if teacher:
                        x0g = model._apply_classifier_free_guidance(x_t, tt, x0, neg_condition=neg)
                if teacher:
                    r[f"Dcfg_{t}"] = float((x - x0g.float()).pow(2).mean())
                    r[f"Dcond_{t}"] = float((x - x0.float()).pow(2).mean())
                else:
                    r[f"D_{t}"] = float((x - x0.float()).pow(2).mean())
            rows.append(r)
            if i % 16 == 0:
                logger.info(f"[{args.dataset} {name} {i + 1}/{n}] " + json.dumps({k: round(v, 4) for k, v in r.items() if k != 'id'}))
        return rows

    def write(path, rows):
        with open(path + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(path + ".tmp", path)

    # ---- check: E0 on the first clips from the fp16 latents vs the stored fp32-latent D_E0
    if args.dataset != "openvid" and args.check_n and not os.path.exists(os.path.join(od, "check_E0.json")):
        model, net, _ = build(MODELS["E0"], False)
        new = d_rows("E0_check", model, net, False, min(args.check_n, N))
        old = [json.loads(l) for l in open(OLD[(args.dataset, "E0")])]
        rel = {f"{t}": max(abs(a[f"D_{t}"] / b[f"D_{t}"] - 1) for a, b in zip(new, old)) for t in T_STUDENT}
        logger.info(f"[{args.dataset} check] E0 max relative |D_t(fp16 lat) / D_t(stored) - 1| = {rel}")
        json.dump({"n": len(new), "max_rel_diff": rel, "new": new}, open(os.path.join(od, "check_E0.json"), "w"), indent=1)
        del model, net
        torch.cuda.empty_cache()

    failed = []
    for name in args.models:
        path = os.path.join(od, f"D_{name}.jsonl")
        if os.path.exists(path):
            logger.info(f"[{args.dataset}] {path} exists; skipping {name}")
            continue
        ckpt = None if name == "teacher" else MODELS[name]
        if ckpt and not os.path.exists(ckpt + ".pth"):
            logger.info(f"[{args.dataset}] MISSING checkpoint {ckpt}; skipping {name}")
            failed.append(name)
            continue
        model, net, _ = build(ckpt, name == "teacher")
        if name == "teacher":
            assert model.config.guidance_scale == 5.0
        write(path, d_rows(name, model, net, name == "teacher", N))
        del model, net
        torch.cuda.empty_cache()
    open(os.path.join(od, "DONE" if not failed else "DONE_PARTIAL"), "w").write(" ".join(failed) + "\n")
    clean_up()


# ----------------------------------------------------------------------------- report (CPU)


def report(args):
    rng = np.random.default_rng(0)
    B = 2000
    keys = {"teacher_cond": ("teacher", "Dcond"), "teacher_cfg": ("teacher", "Dcfg")} | {m: (m, "D") for m in MODELS}
    data = {}
    for ds in DATASETS:
        rows = {}
        for m in ["teacher", *MODELS]:
            p = os.path.join(args.out, ds, f"D_{m}.jsonl")
            p = p if os.path.exists(p) else OLD.get((ds, m), "")
            if os.path.exists(p):
                rows[m] = [json.loads(l) for l in open(p)][: args.n]
        if not rows:
            continue
        n = min(len(v) for v in rows.values())
        ids = [r["id"] for r in next(iter(rows.values()))[:n]]
        for m, v in rows.items():
            assert [r["id"] for r in v[:n]] == ids, f"{ds} {m}: clip order differs"
        pair = np.load(PAIR[ds])[:n, :n]
        boot = [rng.integers(0, n, n) for _ in range(B)]

        def shuf(idx):
            p = pair[np.ix_(idx, idx)]
            return p[idx[:, None] != idx[None, :]].mean()

        data[ds] = dict(rows=rows, n=n, pair=pair, boot=boot, Dshuf=shuf(np.arange(n)),
                        bshuf=np.array([shuf(b) for b in boot]))

    def vals(ds, pred, t):
        m, k = keys[pred]
        if m not in data[ds]["rows"]:
            return None
        return np.array([r[f"{k}_{t}"] for r in data[ds]["rows"][m][: data[ds]["n"]]])

    preds = list(keys)
    res, lines = {}, []
    lines += ["## rho_t = D_t / D_shuf per dataset (95 % CI: clip bootstrap, D_shuf resampled with the clips)", "",
              "D_shuf: " + ", ".join(f"{ds} {data[ds]['Dshuf']:.4f} (n={data[ds]['n']})" for ds in data), "",
              "| x0 predictor | t | " + " | ".join(f"{ds} rho" for ds in data) + " |",
              "|---|---|" + "---|" * len(data)]
    boots = {}
    for p in preds:
        for t in T_STUDENT:
            cells = []
            for ds in data:
                v = vals(ds, p, t)
                if v is None:
                    cells.append("–")
                    continue
                rho = v.mean() / data[ds]["Dshuf"]
                bs = np.array([v[b].mean() for b in data[ds]["boot"]]) / data[ds]["bshuf"]
                boots[(ds, p, t)] = bs
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                res[f"rho_{ds}_{p}_{t}"] = dict(D=float(v.mean()), rho=float(rho), ci=[lo, hi])
                cells.append(f"{rho:.4f} [{lo:.4f}, {hi:.4f}]")
            lines.append(f"| {p} | {t} | " + " | ".join(cells) + " |")

    lines += ["", "## Dataset ratios of rho (independent bootstraps; * = 95 % CI excludes 1)", "",
              "| x0 predictor | t | OpenVid/ViPE | OpenVid/Mixkit | Mixkit/ViPE |", "|---|---|---|---|---|"]
    for p in preds:
        for t in T_STUDENT:
            cells = []
            for a, b in [("openvid", "vipe"), ("openvid", "mixkit"), ("mixkit", "vipe")]:
                if (a, p, t) not in boots or (b, p, t) not in boots:
                    cells.append("–")
                    continue
                r = res[f"rho_{a}_{p}_{t}"]["rho"] / res[f"rho_{b}_{p}_{t}"]["rho"]
                lo, hi = np.nanpercentile(boots[(a, p, t)] / boots[(b, p, t)], [2.5, 97.5])
                res[f"ratio_{a}/{b}_{p}_{t}"] = dict(ratio=r, ci=[lo, hi])
                cells.append(f"{r:.3f}{'*' if lo > 1 or hi < 1 else ''} [{lo:.3f}, {hi:.3f}]")
            lines.append(f"| {p} | {t} | " + " | ".join(cells) + " |")

    # model contrasts within a dataset: same clips / eps / t, so rho_A / rho_B = D_A / D_B (paired clip bootstrap)
    contrasts = [("REF", "E0"), ("E1a_l40s_it40", "E0"), ("E2_l40s_it40", "E0"), ("E1dmd_l40s_it40", "E0"),
                 ("E2_l40s_it40", "E1a_l40s_it40"), ("E1a_l40s_it40", "E1dmd_l40s_it40"),
                 ("E0", "teacher_cond"), ("teacher_cfg", "teacher_cond")]
    lines += ["", "## Model contrasts within a dataset: rho_A / rho_B (= D_A / D_B; paired clip bootstrap)", "",
              "| A / B | t | " + " | ".join(data) + " |", "|---|---|" + "---|" * len(data)]
    for a, b in contrasts:
        for t in T_STUDENT:
            cells = []
            for ds in data:
                va, vb = vals(ds, a, t), vals(ds, b, t)
                if va is None or vb is None:
                    cells.append("–")
                    continue
                r = va.mean() / vb.mean()
                lo, hi = np.percentile([va[i].mean() / vb[i].mean() for i in data[ds]["boot"]], [2.5, 97.5])
                res[f"contrast_{ds}_{a}/{b}_{t}"] = dict(ratio=r, ci=[lo, hi])
                cells.append(f"{r:.3f}{'*' if lo > 1 or hi < 1 else ''} [{lo:.3f}, {hi:.3f}]")
            lines.append(f"| {a} / {b} | {t} | " + " | ".join(cells) + " |")
    json.dump(res, open(os.path.join(args.out, "report.json"), "w"), indent=1, default=float)
    open(os.path.join(args.out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--models", nargs="*", default=[], help=f"teacher and/or {list(MODELS)}")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--check_n", type=int, default=16, help="ViPE / Mixkit: recompute E0 on this many clips (0 = off)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", action="store_true")
    # task2.md B2 (default none): register more student checkpoints, NAME=PATH (repeatable), e.g. the Phase B runs
    ap.add_argument("--extra_model", action="append", default=[], help="NAME=checkpoint path (adds to MODELS)")
    known, _ = ap.parse_known_args()
    for s in known.extra_model:
        name, path = s.split("=", 1)
        assert name not in MODELS and name != "teacher", f"model name {name} already exists"
        MODELS[name] = path
    if "--report" in sys.argv:
        report(ap.parse_args())
        return
    from fastgen.utils.scripts import parse_args

    run(parse_args(ap))


if __name__ == "__main__":
    main()
