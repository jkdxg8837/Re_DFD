# SPDX-License-Identifier: Apache-2.0
"""Why is DFD's delta_sq smaller on Mixkit (real) than on ViPE (synthetic)? Offline measurement, user 2026-10-08.

delta = mean (x - G)^2 exactly as in DFD's student update (official code path, no E3/E4 flags): G = one-step x0
prediction of the student from x_t = forward_process(x, eps, t_student), t_student in t_list[:4] (0.999, 0.937,
0.833, 0.624), conditioned on the clip's caption. Students: the official checkpoints E0 (dmd2_t2v_vipe/0001000, the
student at DFD's first update) and REF (ours_wan_t2v_vipe/0000040, the official DFD result).

One process per dataset (ViPE `long` / Mixkit v3), N clips each:
  1. D_t per clip for each t_student (eps fixed per (clip index, t index) by a CPU generator, so both datasets use
     the same noise); D_shuf = E mean (x - x')^2 over all pairs i != j of the same dataset (pair matrix saved for
     the bootstrap); rho_t = D_t / D_shuf (computed in --report).
  2. Data statistics (no model): E x^2, latent frame-difference energy E (x_{f+1} - x_f)^2, spatial / temporal power
     spectra of the latent, fraction of clips with black borders and of near-static clips (pixel space).
  3. D_0.999 with the empty caption "".
  4. Fresh-noise 4-step student samples (z fixed per index, shared by both datasets) with the dataset's captions;
     I3D features (StyleGAN-V torchscript, 16 frames = every 5th of 81, 224x224) of samples and real clips -> FVD and
     k-NN precision / recall per dataset (in --report).
Latents: VAE posterior mean of frames 0:81 after the training transform (resize small side, center crop 832x480).

Usage (from the repo root):
  torchrun --nproc_per_node=1 --master_port=P scripts/diag/delta_scale.py --dataset vipe|mixkit --n 256 --out DIR \
      --config fastgen/configs/experiments/WanT2V/config_dmd2.py - trainer.fsdp=True trainer.ddp=False log_config.name=X
  python scripts/diag/delta_scale.py --report --out DIR      (numpy / scipy only)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile

import av
import numpy as np

CKPTS = {
    "E0": "checkpoints/wan_t2v_checkpoints_release/dmd2_t2v_vipe/0001000",
    "REF": "checkpoints/wan_t2v_checkpoints_release/ours_wan_t2v_vipe/0000040",
}
I3D = "checkpoints/fvd/i3d_torchscript.pt"
T_STUDENT = [0.999, 0.937, 0.833, 0.624]  # config_dmd2 t_list without the final 0.0
I3D_FRAMES = list(range(0, 80, 5))  # 16 frames


def clip_source(dataset: str, n: int):
    """Yield (id, mp4 bytes, caption) for the first n clips of the dataset's fixed random subset."""
    if dataset == "vipe":  # random 1,280-clip subset of ViPE long (seed 0), scripts/data_prep/prep_vipe_inv.py
        caps = {r["id"]: r["caption"] for r in map(json.loads, open("data/vipe_inv_prep/captions.jsonl"))}
        ids = [l.strip() for l in open("data/vipe_inv_prep/subset1280.txt") if l.strip()][:n]
        for cid in ids:
            yield cid, open(f"data/vipe_inv_prep/clips/{cid}.mp4", "rb").read(), caps[cid]
    elif dataset == "mixkit":  # random 1,280-clip subset of the E2 pool (v3 captions), shuffled into the WDS
        k = 0
        for s in range(13):
            with tarfile.open(f"data/mixkit_inv_wds_v3/{s:06d}.tar") as tf:
                names = sorted(m for m in tf.getnames() if m.endswith(".mp4"))
                for m in names:
                    if k == n:
                        return
                    key = m[:-4]
                    meta = json.loads(tf.extractfile(f"{key}.json").read())
                    yield meta["id"], tf.extractfile(m).read(), tf.extractfile(f"{key}.txt").read().decode().strip()
                    k += 1
    else:
        raise ValueError(dataset)


def decode(mp4: bytes) -> np.ndarray:
    with av.open(io.BytesIO(mp4)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])  # [T, H, W, 3] uint8


def pixel_stats(px):
    """px: [3, T, H, W] in [-1, 1]. Black-border rows/cols and mean absolute frame difference (in [0, 1] units)."""
    v = (px.float() + 1) / 2
    luma = 0.299 * v[0] + 0.587 * v[1] + 0.114 * v[2]  # [T, H, W]
    row = luma.amax(dim=(0, 2))  # brightest value over time and width, per row
    col = luma.amax(dim=(0, 1))
    dark = 0.08  # ~20/255: a row/col never brighter than this in any frame counts as border

    def run(m):  # length of the dark run from the start
        k = 0
        while k < len(m) and m[k]:
            k += 1
        return k

    r, c = (row < dark).tolist(), (col < dark).tolist()
    H, W = len(r), len(c)
    top, bot, left, right = run(r), run(r[::-1]), run(c), run(c[::-1])
    mad = float((v[:, 1:] - v[:, :-1]).abs().mean())
    return {"border_tb": (top + bot) / H, "border_lr": (left + right) / W, "mad": mad}


def latent_stats(x):
    """x: [C, T, H, W] float. Energies and spectra of one latent."""
    import torch

    e2 = float(x.pow(2).mean())
    var = float(x.var())
    fd = float((x[:, 1:] - x[:, :-1]).pow(2).mean())
    # spatial power spectrum (per channel/frame, mean removed per channel/frame), radial bins of |k| / Nyquist
    xs = x - x.mean(dim=(2, 3), keepdim=True)
    P = torch.fft.fft2(xs).abs().pow(2).mean(dim=(0, 1))  # [H, W]
    H, W = P.shape
    ky = torch.fft.fftfreq(H, device=x.device).abs() / 0.5
    kx = torch.fft.fftfreq(W, device=x.device).abs() / 0.5
    kr = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).clamp(max=1.0)
    edges = torch.linspace(0, 1, 11, device=x.device)
    radial = [float(P[(kr >= edges[i]) & ((kr < edges[i + 1]) if i < 9 else (kr <= 1))].sum()) for i in range(10)]
    # temporal power spectrum over the latent frames (mean over channels / pixels)
    xt = x - x.mean(dim=1, keepdim=True)
    Pt = torch.fft.rfft(xt, dim=1).abs().pow(2).mean(dim=(0, 2, 3))
    return {"e2": e2, "var": var, "mean": float(x.mean()), "framediff": fd,
            "spatial_radial": radial, "temporal": Pt.tolist()}


def run_dataset(args):
    import torch

    import fastgen.utils.logging_utils as logger
    from fastgen.datasets.wds_dataloaders import transform_video
    from fastgen.utils import basic_utils
    from fastgen.utils.distributed import clean_up
    from fastgen.utils.scripts import setup
    from scripts.inference.inference_utils import (cleanup_unused_modules, init_checkpointer, init_model,
                                                   load_checkpoint, setup_inference_modules)

    config = setup(args, evaluation=True)
    out = os.path.join(args.out, args.dataset)
    os.makedirs(out, exist_ok=True)
    i3d = torch.jit.load(I3D, map_location="cpu").eval().to("cuda")

    def i3d_feat(px):  # px [3, T, H, W] in [-1, 1] -> 400-d feature
        v = ((px[:, I3D_FRAMES].float().clamp(-1, 1) + 1) * 127.5).round().unsqueeze(0).contiguous()
        with torch.no_grad():
            return i3d(v.to("cuda"), rescale=True, resize=True, return_features=True)[0].cpu().numpy()

    def build(ckpt):
        model = init_model(config)
        it, _ = load_checkpoint(init_checkpointer(config), model, ckpt, config)
        assert it is not None, f"checkpoint not loaded: {ckpt}"
        cleanup_unused_modules(model, False)
        _, student, vae = setup_inference_modules(model, config, False, True, model.precision)
        return model, student, vae

    T = config.model.input_shape[1]
    seq = (T - 1) * 4 + 1
    W, H = config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8
    lat, lat32, conds, ids, rec = [], [], [], [], []
    real_i3d = []
    model, student, vae = build(CKPTS["E0"])
    ctx = {"dtype": model.precision, "device": model.device}

    def encode_text(caps):
        with basic_utils.inference_mode(model.net.text_encoder, precision_amp=model.precision_amp_enc,
                                        device_type="cuda"):
            return basic_utils.to(model.net.text_encoder.encode(caps), **ctx)

    # ---- data pass: latents, captions, pixel / latent statistics, real I3D features
    for n, (cid, mp4, cap) in enumerate(clip_source(args.dataset, args.n)):
        video = torch.from_numpy(decode(mp4))
        px = transform_video(video, seq, (W, H))["real"]  # [3, T, H, W] in [-1, 1]
        with basic_utils.inference_mode(vae, precision_amp=model.precision_amp_enc, device_type="cuda"):
            x = model.net.vae.encode(px.unsqueeze(0).to(**ctx), mode="argmax").float()[0]
        r = {"id": cid, "nframes_src": int(video.shape[0]), "src_hw": list(video.shape[1:3]), **pixel_stats(px)}
        r.update(latent_stats(x))
        rec.append(r)
        real_i3d.append(i3d_feat(px))
        lat.append(x.half().cpu())
        lat32.append(x.cpu())
        conds.append(encode_text([cap]).cpu())
        ids.append(cid)
        if n % 32 == 0:
            logger.info(f"[{args.dataset} data {n + 1}/{args.n}] {cid} e2={r['e2']:.3f} fd={r['framediff']:.3f}")
    N = len(ids)
    empty = encode_text([""]).cpu()
    # all-pairs mean squared distance (D_shuf), on GPU in fp32
    X = torch.stack(lat32).to("cuda").flatten(1)
    sq = X.pow(2).sum(1)
    G = X @ X.T
    pair = ((sq[:, None] + sq[None, :] - 2 * G) / X.shape[1]).clamp_min(0).cpu().numpy()
    del X, G
    np.save(os.path.join(out, "pair_msd.npy"), pair)
    np.save(os.path.join(out, "i3d_real.npy"), np.stack(real_i3d))
    torch.save({"ids": ids, "lat": torch.stack(lat)}, os.path.join(out, "latents_fp16.pt"))
    with open(os.path.join(out, "clips.jsonl"), "w") as f:
        for r in rec:
            f.write(json.dumps(r) + "\n")

    # ---- model passes (steps 1, 3, 4)
    for name in ["E0", "REF"]:
        if name != "E0":
            del model, student, vae
            torch.cuda.empty_cache()
            model, student, vae = build(CKPTS[name])
        ns = model.net.noise_scheduler

        def x0_pred(x_t, t, cond):
            tt = torch.full((1,), t, device="cuda", dtype=ns.t_precision)
            with basic_utils.inference_mode(student, precision_amp=model.precision_amp_infer, device_type="cuda"):
                return student(x_t.to(model.precision), tt, condition=cond, fwd_pred_type="x0").float()

        rows, gen_i3d = [], []
        for i in range(N):
            x = lat32[i].to("cuda").unsqueeze(0)
            cond = conds[i].to("cuda")
            r = {"id": ids[i]}
            for k, t in enumerate(T_STUDENT):
                eps = torch.randn(x.shape, generator=torch.Generator().manual_seed(1000 * i + k)).to("cuda")
                x_t = ns.forward_process(x, eps, torch.full((1,), t, device="cuda", dtype=ns.t_precision))
                r[f"D_{t}"] = float((x - x0_pred(x_t, t, cond)).pow(2).mean())
                if k == 0:  # step 3: empty caption, same x_t
                    r[f"D_{t}_empty"] = float((x - x0_pred(x_t, t, empty.to("cuda"))).pow(2).mean())
            # step 4: fresh-noise sample with this caption (z shared across datasets by index)
            z = torch.randn((1, *config.model.input_shape), generator=torch.Generator().manual_seed(10**6 + i))
            g = model.generator_fn(student, z.to(**ctx), condition=cond, neg_condition=None,
                                   student_sample_steps=model.config.student_sample_steps,
                                   student_sample_type=model.config.student_sample_type,
                                   t_list=model.config.sample_t_cfg.t_list, precision_amp=model.precision_amp_infer)
            with basic_utils.inference_mode(vae, precision_amp=model.precision_amp_enc, device_type="cuda"):
                gpx = model.net.vae.decode(g.to(model.precision)).float().clamp(-1, 1)[0]
            gen_i3d.append(i3d_feat(gpx))
            r["gen_e2"] = float(g.float().pow(2).mean())
            rows.append(r)
            if i % 16 == 0:
                logger.info(f"[{args.dataset} {name} {i + 1}/{N}] " + json.dumps({k: round(v, 4) for k, v in r.items() if k != 'id'}))
        with open(os.path.join(out, f"D_{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        np.save(os.path.join(out, f"i3d_gen_{name}.npy"), np.stack(gen_i3d))
    with open(os.path.join(out, "DONE"), "w") as f:
        f.write("ok\n")
    clean_up()


# ----------------------------------------------------------------------------- report (CPU)


def fvd(a, b):
    from scipy import linalg

    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs, _ = linalg.sqrtm(s1 @ s2, disp=False)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * np.real(cs)))


def prec_rec(real, gen, k=3):
    """Improved precision / recall (Kynkaenniemi et al. 2019) with k-NN radii."""
    def radii(f):
        d = np.linalg.norm(f[:, None] - f[None], axis=-1)
        return np.sort(d, axis=1)[:, k]

    rr, rg = radii(real), radii(gen)
    d = np.linalg.norm(gen[:, None] - real[None], axis=-1)  # [gen, real]
    precision = float((d <= rr[None]).any(1).mean())
    recall = float((d.T <= rg[None]).any(1).mean())
    return precision, recall


def report(args):
    rng = np.random.default_rng(0)
    B = 2000
    out = {}
    lines = []
    data = {}
    for ds in ["vipe", "mixkit"]:
        d = os.path.join(args.out, ds)
        clips = [json.loads(l) for l in open(os.path.join(d, "clips.jsonl"))]
        pair = np.load(os.path.join(d, "pair_msd.npy"))
        D = {m: [json.loads(l) for l in open(os.path.join(d, f"D_{m}.jsonl"))] for m in CKPTS}
        data[ds] = dict(clips=clips, pair=pair, D=D, real=np.load(os.path.join(d, "i3d_real.npy")),
                        gen={m: np.load(os.path.join(d, f"i3d_gen_{m}.npy")) for m in CKPTS})

    def shuf(pair, idx):
        p = pair[np.ix_(idx, idx)]
        n = len(idx)
        same = idx[:, None] == idx[None, :]
        return p[~same].mean() if n > 1 else np.nan

    # step 1
    lines += ["## Step 1: rho_t = D_t / D_shuf", "",
              "| student | t_student | ViPE D_t | ViPE rho_t | Mixkit D_t | Mixkit rho_t | rho Mixkit/ViPE (95% CI) |",
              "|---|---|---|---|---|---|---|"]
    res1 = {}
    for ds in data:
        n = len(data[ds]["clips"])
        data[ds]["boot"] = [rng.integers(0, n, n) for _ in range(B)]
        data[ds]["Dshuf"] = shuf(data[ds]["pair"], np.arange(n))
    for m in CKPTS:
        for t in T_STUDENT + ["0.999_empty"]:
            key = f"D_{t}"
            row = {}
            for ds in data:
                v = np.array([r[key] for r in data[ds]["D"][m]])
                rho = v.mean() / data[ds]["Dshuf"]
                bs = np.array([v[b].mean() / shuf(data[ds]["pair"], b) for b in data[ds]["boot"]])
                row[ds] = dict(D=float(v.mean()), rho=float(rho), boot=bs)
            ratio = row["mixkit"]["rho"] / row["vipe"]["rho"]
            rb = row["mixkit"]["boot"] / row["vipe"]["boot"]
            lo, hi = np.nanpercentile(rb, [2.5, 97.5])
            res1[f"{m}_{t}"] = {ds: {k: row[ds][k] for k in ("D", "rho")} for ds in row} | {"ratio": ratio, "ci": [lo, hi]}
            lines.append(f"| {m} | {t} | {row['vipe']['D']:.4f} | {row['vipe']['rho']:.4f} | {row['mixkit']['D']:.4f} | "
                         f"{row['mixkit']['rho']:.4f} | {ratio:.3f} [{lo:.3f}, {hi:.3f}] |")
    lines += ["", f"D_shuf: ViPE {data['vipe']['Dshuf']:.4f}, Mixkit {data['mixkit']['Dshuf']:.4f} "
              f"(ratio {data['mixkit']['Dshuf'] / data['vipe']['Dshuf']:.3f})", ""]
    out["step1"] = res1
    out["Dshuf"] = {ds: float(data[ds]["Dshuf"]) for ds in data}

    # step 2
    lines += ["## Step 2: data statistics", "", "| stat | ViPE | Mixkit | Mixkit/ViPE |", "|---|---|---|---|"]
    res2 = {}

    def agg(ds, f):
        return float(np.mean([f(c) for c in data[ds]["clips"]]))

    def band(c, lo, hi):  # share of spatial power in radial bins [lo, hi) of 10
        s = np.array(c["spatial_radial"])
        return s[lo:hi].sum() / s.sum()

    def tband(c):  # share of temporal power at nonzero frequencies above the lowest one
        p = np.array(c["temporal"])
        return p[2:].sum() / p[1:].sum()

    stats = [("E x^2 (latent)", lambda c: c["e2"]), ("Var x (latent, per clip)", lambda c: c["var"]),
             ("mean x (latent)", lambda c: c["mean"]),
             ("E (x_{f+1}-x_f)^2 (latent)", lambda c: c["framediff"]),
             ("frame-diff / (2 Var x)", lambda c: c["framediff"] / (2 * c["var"])),
             ("spatial power share, |k| < 0.1 Nyquist", lambda c: band(c, 0, 1)),
             ("spatial power share, |k| >= 0.5 Nyquist", lambda c: band(c, 5, 10)),
             ("temporal power share above lowest freq", tband),
             ("pixel mean |frame diff| ([0,1])", lambda c: c["mad"]),
             ("clips with black border (top+bottom or left+right >= 2 %)",
              lambda c: float(c["border_tb"] >= 0.02 or c["border_lr"] >= 0.02)),
             ("near-static clips (pixel mean |frame diff| < 0.01)", lambda c: float(c["mad"] < 0.01))]
    for name, f in stats:
        v, w = agg("vipe", f), agg("mixkit", f)
        res2[name] = {"vipe": v, "mixkit": w}
        lines.append(f"| {name} | {v:.4g} | {w:.4g} | {w / v if v else float('nan'):.3f} |")
    out["step2"] = res2
    lines.append("")

    # step 4
    lines += ["## Step 4: fresh-noise student samples vs real clips (I3D, 16 frames)", "",
              "| student | dataset | FVD (95% CI) | precision | recall | real split-half FVD |", "|---|---|---|---|---|---|"]
    res4 = {}
    for m in CKPTS:
        for ds in data:
            r, g = data[ds]["real"], data[ds]["gen"][m]
            f = fvd(r, g)
            bs = [fvd(r[b], g[b]) for b in data[ds]["boot"][:200]]
            p, rc = prec_rec(r, g)
            h = len(r) // 2
            perm = rng.permutation(len(r))
            split = fvd(r[perm[:h]], r[perm[h:]])
            res4[f"{m}_{ds}"] = dict(fvd=f, ci=list(np.percentile(bs, [2.5, 97.5])), precision=p, recall=rc,
                                     split_half=split)
            lines.append(f"| {m} | {ds} | {f:.1f} [{np.percentile(bs, 2.5):.1f}, {np.percentile(bs, 97.5):.1f}] | "
                         f"{p:.3f} | {rc:.3f} | {split:.1f} |")
    out["step4"] = res4
    json.dump(out, open(os.path.join(args.out, "report.json"), "w"), indent=1, default=float)
    open(os.path.join(args.out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vipe", "mixkit"])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", action="store_true")
    if "--report" in sys.argv:
        report(ap.parse_args())
        return
    from fastgen.utils.scripts import parse_args

    run_dataset(parse_args(ap))


if __name__ == "__main__":
    main()
