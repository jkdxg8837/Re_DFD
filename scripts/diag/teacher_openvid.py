# SPDX-License-Identifier: Apache-2.0
"""Follow-ups to scripts/diag/delta_scale.py (user, 2026-10-09).

--mode teacher --dataset vipe|mixkit
    rho_t with the teacher instead of the student: same 256 latents (runs/delta_diag/n256/<ds>/latents_fp16.pt),
    same eps per (clip, t) and same 4 t as delta_scale.py; teacher Wan2.1-1.3B x0 at x_t (a) with CFG w=5 and the
    training negative prompt (neg_prompt_wan), exactly DMD2Model._dfd_guided_teacher_x0, and (b) conditional only.
    -> <delta_out>/<ds>/D_teacher.jsonl
--mode openvid
    OpenVidHD (data/openvid_prep, 256 clips, own captions): data statistics as delta_scale.py, I3D features of the
    real clips, and of 4-step fresh-noise samples (z seed 10**6 + i, as delta_scale.py) with the OpenVid captions
    from E0, E1a_seed1_l40s/0000040 and E2_seed1_l40s/0000040. -> <out>/openvid/
--report --out DIR --delta_out DIR   (numpy / scipy)

Usage: torchrun --nproc_per_node=1 scripts/diag/teacher_openvid.py --mode ... --out DIR --delta_out DIR \
           --config fastgen/configs/experiments/WanT2V/config_dmd2.py - trainer.fsdp=True trainer.ddp=False ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from scripts.diag.delta_scale import CKPTS, T_STUDENT, I3D, I3D_FRAMES, clip_source, decode, fvd, latent_stats, \
    pixel_stats, prec_rec

OV_MODELS = {
    "E0": CKPTS["E0"],
    "E1a_l40s_it40": "FASTGEN_OUTPUT/fastgen/wan_dmd2/E1a_seed1_l40s/checkpoints/0000040",
    "E2_l40s_it40": "FASTGEN_OUTPUT/fastgen/wan_dmd2/E2_seed1_l40s/checkpoints/0000040",
}


def openvid_source(n):
    caps = {r["id"]: r["caption"] for r in map(json.loads, open("data/openvid_prep/captions.jsonl"))}
    ids = [l.strip() for l in open("data/openvid_prep/subset256.txt") if l.strip()][:n]
    for cid in ids:
        yield cid, open(f"data/openvid_prep/clips/{cid}.mp4", "rb").read(), caps[cid]


def run(args):
    import torch

    import fastgen.utils.logging_utils as logger
    from fastgen.datasets.wds_dataloaders import PRESET_CONSTANTS, transform_video
    from fastgen.utils import basic_utils
    from fastgen.utils.distributed import clean_up
    from fastgen.utils.scripts import setup
    from scripts.inference.inference_utils import (cleanup_unused_modules, init_checkpointer, init_model,
                                                   load_checkpoint, setup_inference_modules)

    config = setup(args, evaluation=True)

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

    if args.mode == "teacher":
        d = os.path.join(args.delta_out, args.dataset)
        saved = torch.load(os.path.join(d, "latents_fp16.pt"))
        src = list(clip_source(args.dataset, min(args.n, len(saved["ids"]))))
        assert [s[0] for s in src] == saved["ids"][: len(src)], "clip order differs from delta_scale run"
        od = os.path.join(args.out, args.dataset)
        os.makedirs(od, exist_ok=True)
        model, teacher, _ = build(None, True)  # teacher weights come from HF Wan2.1-1.3B, as in training
        assert model.config.guidance_scale == 5.0
        ns = model.net.noise_scheduler
        neg = encode_text(model, [PRESET_CONSTANTS["neg_prompt_wan"]])
        rows = []
        for i, (cid, _, cap) in enumerate(src):
            x = saved["lat"][i].to("cuda").float().unsqueeze(0)
            cond = encode_text(model, [cap])
            r = {"id": cid}
            for k, t in enumerate(T_STUDENT):
                eps = torch.randn(x.shape, generator=torch.Generator().manual_seed(1000 * i + k)).to("cuda")
                tt = torch.full((1,), t, device="cuda", dtype=ns.t_precision)
                x_t = ns.forward_process(x, eps, tt).to(model.precision)
                with basic_utils.inference_mode(teacher, precision_amp=model.precision_amp_infer, device_type="cuda"):
                    x0c = teacher(x_t, tt, condition=cond, fwd_pred_type="x0")
                    x0g = model._apply_classifier_free_guidance(x_t, tt, x0c, neg_condition=neg)  # = training target
                r[f"Dcfg_{t}"] = float((x - x0g.float()).pow(2).mean())
                r[f"Dcond_{t}"] = float((x - x0c.float()).pow(2).mean())
            rows.append(r)
            if i % 16 == 0:
                logger.info(f"[teacher {args.dataset} {i + 1}/{len(src)}] " + json.dumps({k: round(v, 4) for k, v in r.items() if k != 'id'}))
        with open(os.path.join(od, "D_teacher.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        open(os.path.join(od, "DONE_teacher"), "w").write("ok\n")
        clean_up()
        return

    # ---- openvid
    out = os.path.join(args.out, "openvid")
    os.makedirs(out, exist_ok=True)
    i3d = torch.jit.load(I3D, map_location="cpu").eval().to("cuda")

    def i3d_feat(px):
        v = ((px[:, I3D_FRAMES].float().clamp(-1, 1) + 1) * 127.5).round().unsqueeze(0).contiguous()
        with torch.no_grad():
            return i3d(v.to("cuda"), rescale=True, resize=True, return_features=True)[0].cpu().numpy()

    T = config.model.input_shape[1]
    seq = (T - 1) * 4 + 1
    W, H = config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8
    models = dict(m.split("=", 1) for m in args.models) if args.models else OV_MODELS
    extra = bool(args.models) and os.path.exists(os.path.join(out, "i3d_real.npy"))
    first = next(iter(models))
    model, student, vae = build(models[first], False)
    ctx = {"dtype": model.precision, "device": model.device}
    ids, caps, rec, real, lat = [], [], [], [], []
    for n, (cid, mp4, cap) in enumerate(openvid_source(args.n)):
        if extra:  # extra model: real-data pass already done; only the captions are needed
            ids.append(cid)
            caps.append(cap)
            continue
        px = transform_video(torch.from_numpy(decode(mp4)), seq, (W, H))["real"]
        with basic_utils.inference_mode(vae, precision_amp=model.precision_amp_enc, device_type="cuda"):
            x = model.net.vae.encode(px.unsqueeze(0).to(**ctx), mode="argmax").float()[0]
        rec.append({"id": cid, **pixel_stats(px), **latent_stats(x)})
        real.append(i3d_feat(px))
        lat.append(x.cpu())
        ids.append(cid)
        caps.append(cap)
        if n % 32 == 0:
            logger.info(f"[openvid data {n + 1}/{args.n}] {cid} e2={rec[-1]['e2']:.3f}")
    if not extra:
        X = torch.stack(lat).to("cuda").flatten(1)
        sq = X.pow(2).sum(1)
        pair = ((sq[:, None] + sq[None, :] - 2 * X @ X.T) / X.shape[1]).clamp_min(0).cpu().numpy()
        del X
        np.save(os.path.join(out, "pair_msd.npy"), pair)
        np.save(os.path.join(out, "i3d_real.npy"), np.stack(real))
        with open(os.path.join(out, "clips.jsonl"), "w") as f:
            for r in rec:
                f.write(json.dumps(r) + "\n")
    for name, ckpt in models.items():
        if name != first:
            del model, student, vae
            torch.cuda.empty_cache()
            model, student, vae = build(ckpt, False)
        gen = []
        for i in range(len(ids)):
            cond = encode_text(model, [caps[i]])
            z = torch.randn((1, *config.model.input_shape), generator=torch.Generator().manual_seed(10**6 + i))
            g = model.generator_fn(student, z.to(**ctx), condition=cond, neg_condition=None,
                                   student_sample_steps=model.config.student_sample_steps,
                                   student_sample_type=model.config.student_sample_type,
                                   t_list=model.config.sample_t_cfg.t_list, precision_amp=model.precision_amp_infer)
            with basic_utils.inference_mode(vae, precision_amp=model.precision_amp_enc, device_type="cuda"):
                gpx = model.net.vae.decode(g.to(model.precision)).float().clamp(-1, 1)[0]
            gen.append(i3d_feat(gpx))
            if i % 32 == 0:
                logger.info(f"[openvid {name} {i + 1}/{len(ids)}]")
        np.save(os.path.join(out, f"i3d_gen_{name}.npy"), np.stack(gen))
    open(os.path.join(out, "DONE" if not extra else f"DONE_{'_'.join(models)}"), "w").write("ok\n")
    clean_up()


def report(args):
    rng = np.random.default_rng(0)
    lines = []
    res = {}
    # teacher rho
    lines += ["## Teacher rho_t = D_t / D_shuf (same latents, eps, t as the student run)", "",
              "| teacher x0 | t | ViPE D_t | ViPE rho | Mixkit D_t | Mixkit rho | rho Mixkit/ViPE (95% CI) | student E0 ratio |",
              "|---|---|---|---|---|---|---|---|"]
    dd = {}
    for ds in ["vipe", "mixkit"]:
        d = os.path.join(args.delta_out, ds)
        T = [json.loads(l) for l in open(os.path.join(args.out, ds, "D_teacher.jsonl"))]
        n = len(T)
        pair = np.load(os.path.join(d, "pair_msd.npy"))[:n, :n]
        dd[ds] = dict(pair=pair, T=T, S=[json.loads(l) for l in open(os.path.join(d, "D_E0.jsonl"))][:n],
                      boot=[rng.integers(0, n, n) for _ in range(2000)])

    def shuf(pair, idx):
        p = pair[np.ix_(idx, idx)]
        return p[idx[:, None] != idx[None, :]].mean()

    for ds in dd:
        dd[ds]["Dshuf"] = shuf(dd[ds]["pair"], np.arange(len(dd[ds]["pair"])))
        dd[ds]["bshuf"] = np.array([shuf(dd[ds]["pair"], b) for b in dd[ds]["boot"]])

    def rho(ds, rows, key):
        v = np.array([r[key] for r in rows])
        return v.mean(), v.mean() / dd[ds]["Dshuf"], np.array([v[b].mean() for b in dd[ds]["boot"]]) / dd[ds]["bshuf"]

    for kind in ["cfg", "cond"]:
        for t in T_STUDENT:
            row = {ds: rho(ds, dd[ds]["T"], f"D{kind}_{t}") for ds in dd}
            srow = {ds: rho(ds, dd[ds]["S"], f"D_{t}") for ds in dd}
            ratio = row["mixkit"][1] / row["vipe"][1]
            lo, hi = np.percentile(row["mixkit"][2] / row["vipe"][2], [2.5, 97.5])
            sratio = srow["mixkit"][1] / srow["vipe"][1]
            res[f"teacher_{kind}_{t}"] = dict(vipe=row["vipe"][:2], mixkit=row["mixkit"][:2], ratio=ratio, ci=[lo, hi],
                                              student_E0_ratio=sratio)
            lines.append(f"| {'CFG w=5' if kind == 'cfg' else 'cond only'} | {t} | {row['vipe'][0]:.4f} | "
                         f"{row['vipe'][1]:.4f} | {row['mixkit'][0]:.4f} | {row['mixkit'][1]:.4f} | "
                         f"{ratio:.3f} [{lo:.3f}, {hi:.3f}] | {sratio:.3f} |")
    lines.append("")

    # openvid
    o = os.path.join(args.out, "openvid")
    real = np.load(os.path.join(o, "i3d_real.npy"))
    names = list(OV_MODELS) + sorted(os.path.basename(f)[8:-4] for f in __import__("glob").glob(os.path.join(o, "i3d_gen_*.npy"))
                                     if os.path.basename(f)[8:-4] not in OV_MODELS)
    gen = {m: np.load(os.path.join(o, f"i3d_gen_{m}.npy")) for m in names}
    N = len(real)
    subs = [rng.choice(N, int(0.75 * N), replace=False) for _ in range(200)]
    lines += ["## OpenVidHD (256 clips, own captions): fresh-noise samples vs real clips (I3D, 16 frames)", "",
              f"Spread: 200 random subsets of {int(0.75 * N)} of {N} clips without replacement (same subsets for every "
              "model; real clip i and sample i share caption i), 2.5-97.5 %.", "",
              "| student | FVD | FVD subset range | precision | recall |", "|---|---|---|---|---|"]
    sub = {}
    for m in names:
        f = fvd(real, gen[m])
        p, r = prec_rec(real, gen[m])
        sub[m] = np.array([[fvd(real[s], gen[m][s]), *prec_rec(real[s], gen[m][s])] for s in subs])
        lo, hi = np.percentile(sub[m][:, 0], [2.5, 97.5])
        res[f"openvid_{m}"] = dict(fvd=f, fvd_sub=[lo, hi], precision=p, recall=r)
        lines.append(f"| {m} | {f:.1f} | [{lo:.1f}, {hi:.1f}] | {p:.3f} | {r:.3f} |")
    lines += ["", "| contrast (subset-paired) | dFVD (2.5-97.5 %) | dprecision | drecall |", "|---|---|---|---|"]
    pairs = [("E2_l40s_it40", "E0"), ("E2_l40s_it40", "E1a_l40s_it40"), ("E1a_l40s_it40", "E0")]
    pairs += [(a, b) for x in names if x not in OV_MODELS for a, b in [(x, "E0"), ("E1a_l40s_it40", x), ("E2_l40s_it40", x)]]
    for a, b in pairs:
        dif = sub[a] - sub[b]
        q = np.percentile(dif, [2.5, 50, 97.5], axis=0)
        res[f"openvid_{a}_minus_{b}"] = q.tolist()
        lines.append(f"| {a} − {b} | {q[1, 0]:+.1f} [{q[0, 0]:+.1f}, {q[2, 0]:+.1f}] | {q[1, 1]:+.3f} [{q[0, 1]:+.3f}, "
                     f"{q[2, 1]:+.3f}] | {q[1, 2]:+.3f} [{q[0, 2]:+.3f}, {q[2, 2]:+.3f}] |")
    # data statistics vs ViPE / Mixkit
    lines += ["", "## OpenVid data statistics (same definitions as delta_scale step 2)", "",
              "| stat | ViPE | Mixkit | OpenVid |", "|---|---|---|---|"]
    clips = {ds: [json.loads(l) for l in open(os.path.join(args.delta_out, ds, "clips.jsonl"))] for ds in dd}
    clips["openvid"] = [json.loads(l) for l in open(os.path.join(o, "clips.jsonl"))]
    pair_ov = np.load(os.path.join(o, "pair_msd.npy"))
    shufs = {"vipe": dd["vipe"]["Dshuf"], "mixkit": dd["mixkit"]["Dshuf"], "openvid": shuf(pair_ov, np.arange(len(pair_ov)))}
    for name, f in [("E x^2 (latent)", lambda c: c["e2"]), ("E (x_{f+1}-x_f)^2 (latent)", lambda c: c["framediff"]),
                    ("pixel mean |frame diff|", lambda c: c["mad"]),
                    ("near-static clips (mad < 0.01)", lambda c: float(c["mad"] < 0.01)),
                    ("black border clips", lambda c: float(c["border_tb"] >= 0.02 or c["border_lr"] >= 0.02))]:
        v = {ds: float(np.mean([f(c) for c in clips[ds]])) for ds in clips}
        res[f"stat_{name}"] = v
        lines.append(f"| {name} | {v['vipe']:.4g} | {v['mixkit']:.4g} | {v['openvid']:.4g} |")
    lines.append(f"| D_shuf | {shufs['vipe']:.4f} | {shufs['mixkit']:.4f} | {shufs['openvid']:.4f} |")
    json.dump(res, open(os.path.join(args.out, "report.json"), "w"), indent=1, default=float)
    open(os.path.join(args.out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teacher", "openvid"])
    ap.add_argument("--dataset", choices=["vipe", "mixkit"])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta_out", default="runs/delta_diag/n256")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--models", nargs="*", default=None, help="openvid: NAME=CKPT ... (extra models; reuses real pass)")
    if "--report" in sys.argv:
        report(ap.parse_args())
        return
    from fastgen.utils.scripts import parse_args

    run(parse_args(ap))


if __name__ == "__main__":
    main()
