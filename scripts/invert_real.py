# SPDX-License-Identifier: Apache-2.0
"""Teacher-ODE inversion of real clips for E4 (spec §3 E4, decisions 2026-09-28).

For each clip: the 81-frame video goes through the training transform (`transform_video`: resize small side to 480,
center crop 832x480, [-1, 1]), is VAE-encoded (posterior mean, mode="argmax"), and its caption is text-encoded.
The deterministic probability-flow ODE of the teacher Wan2.1-T2V-1.3B (rectified flow, v = eps - x0,
dx/dt = v) is integrated **forward** from t=0 to t=max_t (0.999) with first-order Euler steps on the teacher's
50-step shifted grid (UniPC scheduler timesteps, shift 5, as in teacher sampling), using the conditional
prediction only (no CFG, w=1). The end state z_inv = x_{0.999} is saved; in training it replaces eps_student
(`model.dfd_inverted_noise_key=z_inv`).

Quality check (--recon N): for the first N clips of each shard, integrate the same grid backward from z_inv
(Euler, same conditional teacher) and report latent MSE vs x0 and pixel PSNR of VAE-decoded reconstructions
vs the input frames (plus the VAE-only PSNR of decode(x0) as the ceiling).

Usage (one process per GPU, from the repo root, venv active):
  torchrun --nproc_per_node=1 --master_port=<port> scripts/invert_real.py --list <ids.txt> --out <dir> \
      --shard k --nshards n [--recon 25] --config fastgen/configs/experiments/WanT2V/config_dmd2.py - <overrides>
Inputs: ids.txt = one Mixkit clip id per line; clips at data/mixkit_prep/clips/<id>.mp4, captions in
data/mixkit_prep/captions_v3.jsonl. Output: <out>/z/<id>.npy (float32 [16, 21, 60, 104]) and <out>/meta_s<k>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os

import av
import numpy as np
import torch

import fastgen.utils.logging_utils as logger
from fastgen.datasets.wds_dataloaders import transform_video
from fastgen.utils import basic_utils
from fastgen.utils.distributed import clean_up
from fastgen.utils.scripts import parse_args, setup
from scripts.inference.inference_utils import cleanup_unused_modules, init_model, setup_inference_modules


def load_clip(path: str) -> torch.Tensor:
    with av.open(path) as c:
        fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    return torch.from_numpy(np.stack(fr))  # [T, H, W, 3] uint8


def time_grid(teacher, num_steps: int, shift: float, max_t: float) -> list[float]:
    """Increasing grid 0 = t_0 < ... < t_N = max_t: the teacher's UniPC sigmas (shift) reversed."""
    sch = teacher.unipc_scheduler
    sch.config.flow_shift = shift
    sch.set_timesteps(num_inference_steps=num_steps, device="cpu")
    sig = sorted({float(s) for s in sch.sigmas.flatten().tolist()})
    sig = [min(s, max_t) for s in sig]
    if sig[0] > 0:
        sig = [0.0] + sig
    sig[-1] = max_t
    out = []
    for s in sig:  # dedupe after clamping
        if not out or s > out[-1] + 1e-9:
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips_dir", default="data/mixkit_prep/clips")
    ap.add_argument("--captions", default="data/mixkit_prep/captions_v3.jsonl")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--recon", type=int, default=0, help="reconstruction check on the first N clips of this shard")
    args = parse_args(ap)
    config = setup(args, evaluation=True)

    ids = [l.strip() for l in open(args.list) if l.strip()][args.shard :: args.nshards]
    caps = {r["id"]: r["caption"] for r in map(json.loads, open(args.captions))}
    os.makedirs(os.path.join(args.out, "z"), exist_ok=True)
    meta_path = os.path.join(args.out, f"meta_s{args.shard}.jsonl")
    done = {json.loads(l)["id"] for l in open(meta_path)} if os.path.exists(meta_path) else set()

    model = init_model(config)
    cleanup_unused_modules(model, True)
    teacher, _, vae = setup_inference_modules(model, config, True, False, model.precision)
    assert teacher is not None and vae is not None
    ctx = {"dtype": model.precision, "device": model.device}
    ns = teacher.noise_scheduler
    grid = time_grid(teacher, args.num_steps, args.shift, float(ns.max_t))
    logger.info(f"inversion grid: {len(grid) - 1} Euler steps, t {grid[0]:.4f} -> {grid[-1]:.4f}; first {grid[:4]}")
    T = config.model.input_shape[1]
    seq = (T - 1) * 4 + 1
    W, H = config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8

    def flow(x, t, cond):
        tt = torch.full((x.shape[0],), t, device=x.device)
        tt = ns.safe_clamp(tt, min=ns.min_t, max=ns.max_t).to(model.precision)
        with basic_utils.inference_mode(teacher, precision_amp=model.precision_amp_infer, device_type=model.device.type):
            v = teacher(x.to(model.precision), tt, condition=cond, r=None, return_features_early=False,
                        feature_indices={}, return_logvar=False)
        return v.float()

    with open(meta_path, "a") as fo:
        for n, cid in enumerate(ids):
            if cid in done:
                continue
            video = load_clip(os.path.join(args.clips_dir, f"{cid}.mp4"))
            real = transform_video(video, seq, (W, H))["real"].unsqueeze(0).to(**ctx)  # [1, 3, T, H, W]
            with basic_utils.inference_mode(vae, precision_amp=model.precision_amp_enc, device_type=model.device.type):
                x0 = model.net.vae.encode(real, mode="argmax").float()
            with basic_utils.inference_mode(model.net.text_encoder, precision_amp=model.precision_amp_enc,
                                            device_type=model.device.type):
                cond = basic_utils.to(model.net.text_encoder.encode([caps[cid]]), **ctx)
            x = x0.clone()
            for k in range(len(grid) - 1):  # forward Euler: x_{t+dt} = x_t + dt * v(x_t, t)
                x = x + (grid[k + 1] - grid[k]) * flow(x, grid[k], cond)
            z = x
            np.save(os.path.join(args.out, "z", f"{cid}.npy"), z[0].cpu().numpy().astype(np.float32))
            rec = {"id": cid, "z_mean": float(z.mean()), "z_std": float(z.std()), "x0_std": float(x0.std()),
                   "corr_z_x0": float(torch.corrcoef(torch.stack([z.flatten(), x0.flatten()]))[0, 1])}
            if n < args.recon:
                y = z.clone()
                for k in range(len(grid) - 1, 0, -1):  # backward Euler from max_t to 0
                    y = y - (grid[k] - grid[k - 1]) * flow(y, grid[k], cond)
                with basic_utils.inference_mode(vae, precision_amp=model.precision_amp_enc, device_type=model.device.type):
                    px_rec = model.net.vae.decode(y.to(model.precision)).float().clamp(-1, 1)
                    px_vae = model.net.vae.decode(x0.to(model.precision)).float().clamp(-1, 1)
                ref = real.float()
                psnr = lambda a: float(10 * torch.log10(4.0 / ((a - ref) ** 2).mean()))  # range [-1, 1] -> peak 2
                rec.update(latent_mse=float(((y - x0) ** 2).mean()), latent_var=float(x0.var()),
                           psnr_recon=psnr(px_rec), psnr_vae=psnr(px_vae))
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            logger.info(f"[{n + 1}/{len(ids)}] {cid} {json.dumps({k: round(v, 4) for k, v in rec.items() if k != 'id'})}")
    clean_up()


if __name__ == "__main__":
    main()
