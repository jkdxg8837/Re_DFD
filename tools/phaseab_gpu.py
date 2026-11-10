# SPDX-License-Identifier: Apache-2.0
"""Phase 0 / A GPU work for task2.md (runs/phaseAB/). One process per GPU.

Identical pipeline for every set (task2.md 0.1): uint8 RGB frames (from mp4) -> training transform
`transform_video` (frames 0:81 as the training dataloader with decoder_type="full", resize small side, center crop
832x480, [-1, 1]) -> VAE `encode(mode="sample")` (training mode; CUDA RNG seeded 3000 + sample index) -> latent
[16, 21, 60, 104], stored fp16. Generated samples are first written to mp4 (libx264 crf 18, 16 fps) and read back, so
they pass through the same codec -> transform -> encode path as the reference clips.

Per sample also:
  stats   decoded pixels of the stored latent (VAE decode): mean luminance (Rec.601), mean HSV saturation, dark-pixel
          ratio (luminance < 0.1), per-channel latent means (16)
  feat_a  per-channel latent mean / std over (T, H, W) (32) + 4x spatially average-pooled latent [16, 21, 15, 26]
  feat_b  frozen teacher features of DMD2's discriminator hook (`feature_indices={15, 22, 29}`), unpatchified
          [1536, 21, 30, 52] per block, pooled to per-channel mean and std over (T, H, W) -> 3 x 3072, at
          t = min_t (0.001, "clean") and t = 0.5 (x_t = forward_process(x, eps, t), eps CPU-seeded 4000 + index);
          condition = empty-string embedding (caption-independent, so the probe cannot read the caption style)

Subcommands (from the repo root, venv .venv):
  torchrun --nproc_per_node=1 --master_port=P tools/phaseab_gpu.py --cmd ref --set R_mx [--n N] --config ... - <overrides>
  ... --cmd gen --gen T_w5|T_w1|A1_w5|A1_w1|dmd2|dfd40|E1a|E2 [--n N]
  ... --cmd gen --gen <tag> --ckpt <student checkpoint prefix>   (any student, standard 4-step eval sampler)
  ... --cmd blur --set R_mx --sigma S      (features of Gaussian-blurred stored latents, task2.md 0.5)
  ... --cmd a3 --ds vipe|mixkit|openvid [--n N]   (per-sample teacher x0_c at t = 0.999 on the shift-report clips)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time

import av
import numpy as np
import torch
import torch.nn.functional as F

from tools.phaseab_util import gaussian_blur_hw

ROOT = os.environ.get("PHASEAB_ROOT", "runs/phaseAB")  # data_prep.md A-1..A-3 redo: runs/phaseAB/v2
FEAT_BLOCKS = {15, 22, 29}
T_FEAT = [None, 0.5]  # None = noise_scheduler.min_t
T_LIST = [0.999, 0.937, 0.833, 0.624, 0.0]
GEN = {  # name -> (kind, checkpoint, guidance)
    "T_w5": ("teacher50", None, 5.0),
    "T_w1": ("teacher50", None, 1.0),
    "A1_w5": ("teacher4", None, 5.0),
    "A1_w1": ("teacher4", None, 1.0),
    "dmd2": ("student", "checkpoints/wan_t2v_checkpoints_release/dmd2_t2v_vipe/0001000", None),
    "dfd40": ("student", "checkpoints/wan_t2v_checkpoints_release/ours_wan_t2v_vipe/0000040", None),
    "E1a": ("student", "FASTGEN_OUTPUT/fastgen/wan_dmd2/E1a_seed1_l40s/checkpoints/0000040", None),
    "E2": ("student", "FASTGEN_OUTPUT/fastgen/wan_dmd2/E2_seed1_l40s/checkpoints/0000040", None),
}


def read_mp4(path_or_bytes) -> torch.Tensor:
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    with av.open(src) as c:
        return torch.from_numpy(np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)]))  # [T,H,W,3]


def write_mp4(frames: np.ndarray, path: str, fps: int = 16):
    """frames uint8 [T, H, W, 3] -> h264 crf 18 yuv420p."""
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        s.options = {"crf": "18", "preset": "medium"}
        for fr in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def to_uint8(px: torch.Tensor) -> np.ndarray:
    """px [1, 3, T, H, W] in [-1, 1] -> uint8 [T, H, W, 3] (same rounding as fastgen save_media)."""
    v = ((px[0].float().clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return v.permute(1, 2, 3, 0).cpu().numpy()


class Worker:
    def __init__(self, config, ckpt=None):
        import fastgen.utils.logging_utils as logger
        from fastgen.datasets.wds_dataloaders import PRESET_CONSTANTS
        from fastgen.utils import basic_utils
        from scripts.inference.inference_utils import (cleanup_unused_modules, init_checkpointer, init_model,
                                                       setup_inference_modules)

        self.logger, self.bu = logger, basic_utils
        self.config = config
        model = init_model(config)
        if ckpt:
            self._load(model, init_checkpointer(config), ckpt)
        cleanup_unused_modules(model, True)
        self.teacher, self.student, self.vae = setup_inference_modules(model, config, True, True, model.precision)
        self.model = model
        self.ns = model.net.noise_scheduler
        self.ctx = {"dtype": model.precision, "device": model.device}
        self.neg = self.encode_text([PRESET_CONSTANTS["neg_prompt_wan"]])
        self.emp = self.encode_text([""])
        T = config.model.input_shape[1]
        self.seq = (T - 1) * 4 + 1
        self.size = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)  # (W, H)
        assert (T, self.seq, self.size) == (21, 81, (832, 480)), (T, self.seq, self.size)
        assert list(config.model.sample_t_cfg.t_list) == T_LIST, config.model.sample_t_cfg.t_list
        assert config.model.student_sample_type == "ode" and config.model.student_sample_steps == 4

    def _load(self, model, checkpointer, ckpt):
        """Load the net (+EMA) like scripts/inference/inference_utils.load_checkpoint and print the load report
        (spec §0 rule 3): DCP loads into the model's own state dict, so keys are compared with the checkpoint
        metadata to report missing / unexpected keys explicitly."""
        import torch.distributed.checkpoint as dcp

        mdict = torch.nn.ModuleDict({"net": model.net, **model.ema_dict})
        for k, m in mdict.items():
            meta = dcp.FileSystemReader(f"{ckpt}.{k}_model").read_metadata()
            ck = set(meta.state_dict_metadata.keys())
            mk = set(m.state_dict().keys())
            missing, unexpected = sorted(mk - ck), sorted(ck - mk)
            n = sum(p.numel() for p in m.parameters())
            dt = sorted({str(p.dtype) for p in m.parameters()})
            self.logger.info(f"[load-report] {k}: ckpt={ckpt} params={n / 1e9:.4f}B dtypes={dt} "
                             f"missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                self.logger.warning(f"[load-report] {k}: first missing: {missing[:5]}")
            if unexpected:
                self.logger.warning(f"[load-report] {k}: first unexpected: {unexpected[:5]}")
        it = checkpointer.load(mdict, path=ckpt)
        assert it is not None and it > 0, f"checkpoint not loaded: {ckpt}"
        self.logger.info(f"[load-report] loaded iteration {it} from {ckpt}")

    def encode_text(self, caps):
        bu, m = self.bu, self.model
        with bu.inference_mode(m.net.text_encoder, precision_amp=m.precision_amp_enc, device_type="cuda"):
            return bu.to(m.net.text_encoder.encode(caps), **self.ctx)

    # ---------------------------------------------------------------- shared per-sample pipeline
    @torch.no_grad()
    def process(self, frames: torch.Tensor, idx: int) -> dict:
        from fastgen.datasets.wds_dataloaders import transform_video

        bu, m = self.bu, self.model
        real = transform_video(frames, self.seq, self.size)["real"].unsqueeze(0).to(**self.ctx)
        torch.manual_seed(3000 + idx)
        with bu.inference_mode(self.vae, precision_amp=m.precision_amp_enc, device_type="cuda"):
            lat = m.net.vae.encode(real, mode="sample").float()
        return self.describe(lat, idx, decode=True)

    @torch.no_grad()
    def describe(self, lat: torch.Tensor, idx: int, decode: bool) -> dict:
        bu, m = self.bu, self.model
        out = {"lat": lat[0].half().cpu()}
        if decode:
            with bu.inference_mode(self.vae, precision_amp=m.precision_amp_enc, device_type="cuda"):
                px = m.net.vae.decode(lat.to(m.precision)).float().clamp(-1, 1)
            v = (px[0] + 1) / 2  # [3, T, H, W] in [0, 1]
            lum = 0.299 * v[0] + 0.587 * v[1] + 0.114 * v[2]
            mx, mn = v.amax(0), v.amin(0)
            sat = torch.where(mx > 0, (mx - mn) / mx.clamp_min(1e-8), torch.zeros_like(mx))
            out["stats"] = np.array([float(lum.mean()), float(sat.mean()), float((lum < 0.1).float().mean())]
                                    + lat[0].mean(dim=(1, 2, 3)).tolist(), dtype=np.float32)
        x = lat[0]
        pooled = F.avg_pool2d(x.reshape(-1, *x.shape[-2:]), 4).reshape(x.shape[0], x.shape[1], 15, 26)
        out["feat_a"] = torch.cat([x.mean(dim=(1, 2, 3)), x.std(dim=(1, 2, 3)), pooled.flatten()]).half().cpu()
        fb = []
        for tf in T_FEAT:
            t = self.ns.min_t if tf is None else tf
            tt = torch.full((1,), t, device="cuda", dtype=self.ns.t_precision)
            if tf is None:
                xt = lat
            else:
                eps = torch.randn(lat.shape, generator=torch.Generator().manual_seed(4000 + idx)).to("cuda")
                xt = self.ns.forward_process(lat, eps, tt)
            with bu.inference_mode(self.teacher, precision_amp=m.precision_amp_infer, device_type="cuda"):
                feats = self.teacher(xt.to(m.precision), tt, condition=self.emp, return_features_early=True,
                                     feature_indices=FEAT_BLOCKS)
            for f in feats:  # [1, 1536, 21, 30, 52], blocks in increasing index order
                f = f.float()[0]
                fb += [f.mean(dim=(1, 2, 3)), f.std(dim=(1, 2, 3))]
        out["feat_b"] = torch.cat(fb).cpu()
        return out

    # ---------------------------------------------------------------- generators
    @torch.no_grad()
    def generate(self, kind, w, noise, cond):
        bu, m = self.bu, self.model
        if kind == "teacher50":
            with bu.inference_mode(self.teacher, precision_amp=m.precision_amp_infer, device_type="cuda"):
                return self.teacher.sample(noise, condition=cond, neg_condition=self.neg,
                                           guidance_scale=None if w == 1.0 else w, num_steps=50, shift=5.0)
        if kind == "student" or (kind == "teacher4" and w == 1.0):
            net = self.student if kind == "student" else self.teacher
            # the standard eval sampler (scripts/inference/video_model_inference.py student path)
            return m.generator_fn(net, noise, condition=cond, neg_condition=self.neg, student_sample_steps=4,
                                  student_sample_type="ode", t_list=T_LIST,
                                  precision_amp=m.precision_amp_infer).to(noise.dtype)
        assert kind == "teacher4"
        # = DMD2Model._student_sample_loop ("ode") with CFG on the x0 prediction at each of the 4 steps:
        # x0 = x0_c + (w - 1)(x0_c - x0_neg), the training-target convention (dmd2.py:203)
        net, ns = self.teacher, self.ns
        tl = torch.tensor(T_LIST, device=noise.device, dtype=ns.t_precision)
        with bu.inference_mode(net, precision_amp=m.precision_amp_infer, device_type="cuda"):
            x = ns.latents(noise=noise, t_init=tl[0])
            for t_cur, t_next in zip(tl[:-1], tl[1:]):
                tb = t_cur.expand(x.shape[0])
                x0 = net(x, tb, condition=cond, fwd_pred_type="x0")
                x0 = x0 + (w - 1) * (x0 - net(x, tb, condition=self.neg, fwd_pred_type="x0"))
                if t_next > 0:
                    eps = ns.x0_to_eps(xt=x, x0=x0, t=tb)
                    x = ns.forward_process(x0, eps, t_next.expand(x.shape[0]))
        return x0.to(noise.dtype)


def save_set(name, rows, extra_meta):
    d = os.path.join(ROOT, "latents", name)
    os.makedirs(d, exist_ok=True)
    torch.save(torch.stack([r["lat"] for r in rows]), os.path.join(d, "latents.pt"))
    for k in ["stats", "feat_a", "feat_b"]:
        if k in rows[0]:
            v = [r[k].numpy() if torch.is_tensor(r[k]) else r[k] for r in rows]
            np.save(os.path.join(d, f"{k}.npy"), np.stack(v))
    with open(os.path.join(d, "meta.jsonl"), "w") as f:
        for r in extra_meta:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "DONE"), "w") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")


def cmd_ref(w: Worker, args):
    rows_in = [json.loads(l) for l in open(os.path.join(ROOT, "sets", f"{args.set}.jsonl"))][: args.n]
    name = args.set if args.n >= 300 else f"{args.set}_n{args.n}"
    out, meta = [], []
    for r in rows_in:
        o = w.process(read_mp4(r["mp4"]), r["idx"])
        out.append(o)
        meta.append({"idx": r["idx"], "id": r["id"], "caption": r["caption"]})
        if r["idx"] % 25 == 0:
            w.logger.info(f"[ref {name}] {r['idx'] + 1}/{len(rows_in)} stats={o['stats'][:3].tolist()}")
    save_set(name, out, meta)


def cmd_gen(w: Worker, args):
    kind, _, gw = GEN[args.gen]
    prompts = [json.loads(l) for l in open(os.path.join(ROOT, "prompts.jsonl"))][: args.n]
    name = args.gen if args.n >= 200 else f"{args.gen}_n{args.n}"
    vd = os.path.join(ROOT, "videos", name)
    os.makedirs(vd, exist_ok=True)
    out, meta = [], []
    shape = [1, *w.config.model.input_shape]
    for p in prompts:
        k = p["prompt_idx"]
        noise = torch.randn(shape, generator=torch.Generator().manual_seed(p["seed"])).to(**w.ctx)
        t0 = time.time()
        lat = w.generate(kind, gw, noise, w.encode_text([p["prompt"]]))
        tg = time.time() - t0
        with w.bu.inference_mode(w.vae, precision_amp=w.model.precision_amp_enc, device_type="cuda"):
            px = w.model.net.vae.decode(lat.to(w.model.precision))
        mp4 = os.path.join(vd, f"p{k:03d}.mp4")
        write_mp4(to_uint8(px), mp4)
        o = w.process(read_mp4(mp4), k)  # identical pipeline as the reference clips
        out.append(o)
        meta.append({"idx": k, "source": p["source"], "prompt": p["prompt"], "seed": p["seed"], "mp4": mp4,
                     "gen_seconds": round(tg, 2)})
        if k % 20 == 0:
            w.logger.info(f"[gen {name}] {k + 1}/{len(prompts)} gen {tg:.1f}s stats={o['stats'][:3].tolist()}")
    save_set(name, out, meta)


def cmd_blur(w: Worker, args):
    d = os.path.join(ROOT, "latents", args.set)
    lat = torch.load(os.path.join(d, "latents.pt"))
    name = f"{args.set}_blur{args.sigma:g}"
    out = []
    for i in range(lat.shape[0]):
        x = gaussian_blur_hw(lat[i : i + 1].to("cuda").float(), args.sigma)
        o = w.describe(x, i, decode=False)
        out.append(o)
    od = os.path.join(ROOT, "latents", name)
    os.makedirs(od, exist_ok=True)
    for k in ["feat_a", "feat_b"]:
        np.save(os.path.join(od, f"{k}.npy"), np.stack([o[k].numpy() for o in out]))
    with open(os.path.join(od, "DONE"), "w") as f:
        f.write(f"sigma={args.sigma} from {d}\n")


def cmd_a3(w: Worker, args):
    """Per-sample conditional teacher x0 at t = 0.999 on the shift-report clips, exactly as scripts/diag/s_score.py
    (eps = CPU randn seeded 1000 * i + T_STUDENT.index(0.999) = 1000 * i, caption = the clip's own). Saves the mean
    residual b_bar = mean_i (x_i - x0_c,i), its Gram <b_i, b_j> / d, D_c per clip (checked against s_score), and the
    top-64 PCA of the clean latents (eigenvalues lambda_k = Var <x, v_k>, projections b_k = <b_bar, v_k>)."""
    from scripts.diag.s_score import LAT, sources

    ds, bu, m = args.ds, w.bu, w.model
    saved = torch.load(LAT[ds])
    src = sources(ds, min(args.n, len(saved["ids"])))
    assert [s[0] for s in src] == saved["ids"][: len(src)]
    N = len(src)
    rows_old = {}
    p_old = f"runs/s_score/n256/{ds}/t0.999/rows.jsonl"
    if os.path.exists(p_old):
        rows_old = {r["vid"]: r for r in map(json.loads, open(p_old))}
    B, D_c = [], []
    t = 0.999
    tt = torch.full((1,), t, device="cuda", dtype=w.ns.t_precision)
    for i, (cid, _, cap) in enumerate(src):
        x = saved["lat"][i].to("cuda").float().unsqueeze(0)
        eps = torch.randn(x.shape, generator=torch.Generator().manual_seed(1000 * i + 0)).to("cuda")
        x_t = w.ns.forward_process(x, eps, tt).to(m.precision)
        cond = w.encode_text([cap])
        with bu.inference_mode(w.teacher, precision_amp=m.precision_amp_infer, device_type="cuda"):
            x0 = w.teacher(x_t, tt, condition=cond, fwd_pred_type="x0").float()
        b = (x - x0)[0]
        D_c.append(float(b.pow(2).mean()))
        B.append(b.flatten().half().cpu())
        if i % 32 == 0:
            o = rows_old.get(cid, {}).get("D_c")
            w.logger.info(f"[a3 {ds}] {i + 1}/{N} D_c={D_c[-1]:.5f} (s_score {o})")
    Bm = torch.stack(B)  # [N, d] fp16 on CPU
    d = Bm.shape[1]
    bbar = Bm.float().mean(0)
    G = torch.zeros(N, N, dtype=torch.float64)
    for i0 in range(0, N, 32):
        bi = Bm[i0 : i0 + 32].to("cuda").float()
        for j0 in range(0, N, 32):
            G[i0 : i0 + 32, j0 : j0 + 32] = (bi @ Bm[j0 : j0 + 32].to("cuda").float().T / d).double().cpu()
    del Bm
    # PCA of the clean latents (same N clips) via the N x N Gram of the centered data
    X = saved["lat"][:N].float().flatten(1)
    mu = X.mean(0)
    Xc = (X - mu).to("cuda")
    K = (Xc @ Xc.T).double()
    ev, U = torch.linalg.eigh(K)
    kk = min(64, N - 1)  # centered data of N clips has rank <= N - 1
    ev, U = ev.flip(0)[:kk], U.flip(1)[:, :kk]
    assert float(ev[-1]) > 0, "non-positive PCA eigenvalue"
    V = (Xc.T.double() @ U) / ev.clamp_min(1e-12).sqrt()  # [d, 64] unit principal directions
    lam = (ev / (N - 1)).cpu()  # Var <x, v_k>
    bk = (V.T @ bbar.to("cuda").double()).cpu()
    od = os.path.join(ROOT, "A3", ds if N >= 256 else f"{ds}_n{N}")
    os.makedirs(od, exist_ok=True)
    torch.save({"bbar": bbar, "mu_x": mu, "gram_b": G, "D_c": torch.tensor(D_c), "ids": [s[0] for s in src],
                "lam": lam, "bk": bk, "bbar_sq_total": float(bbar.double().pow(2).sum()), "d": d,
                "bbar_in_pca_sq": float(bk.pow(2).sum())}, os.path.join(od, "a3.pt"))
    w.logger.info(f"[a3 {ds}] done N={N}, ||b_bar||^2/d={float(bbar.pow(2).mean()):.5f}, "
                  f"share in top-64 PCA {float(bk.pow(2).sum()) / float(bbar.double().pow(2).sum()):.3f}")


def main():
    from fastgen.utils.distributed import clean_up
    from fastgen.utils.scripts import parse_args, setup

    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", required=True, choices=["ref", "gen", "blur", "a3"])
    ap.add_argument("--set")
    ap.add_argument("--gen")
    ap.add_argument("--ds")
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=10**9)
    ap.add_argument("--ckpt", help="gen: any student checkpoint prefix; --gen is then a free tag (Phase B reuse)")
    args = parse_args(ap)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = setup(args, evaluation=True)
    if args.cmd == "gen" and args.ckpt:
        assert args.gen not in GEN or GEN[args.gen][1] == args.ckpt, f"tag {args.gen} is reserved"
        GEN[args.gen] = ("student", args.ckpt, None)
    ckpt = GEN[args.gen][1] if args.cmd == "gen" else None
    w = Worker(config, ckpt)
    {"ref": cmd_ref, "gen": cmd_gen, "blur": cmd_blur, "a3": cmd_a3}[args.cmd](w, args)
    clean_up()


if __name__ == "__main__":
    main()
