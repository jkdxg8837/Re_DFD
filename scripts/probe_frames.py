# SPDX-License-Identifier: Apache-2.0
"""Memory probe for the latent frame count T (spec §2.5 of `Dfd experiments.md`).

Driver mode (default) walks the frame ladder from the largest T down. For each T it first tries per-GPU batch 2,
then per-GPU batch 1 (a semantics-preserving saving: the global batch stays 16, FastGen recomputes grad
accumulation). It stops at the first (T, batch) whose peak `torch.cuda.max_memory_allocated()` stays within
`--max_frac` of GPU memory on every rank. Each trial is a real `train.py` run of the experiment config for
`--max_iter` iterations. With `student_update_freq=5` that covers critic steps (iters 1-4, 6-9) and student
steps (iters 5, 10).

Worker mode (`--probe_out` set; launched by the driver through torchrun) runs exactly what `train.py` runs,
with two probe-only changes that do not touch repo code:
  * `Trainer.save_checkpoint` is a no-op, so the probe never writes ~17 GB checkpoints.
  * Each rank writes `rank<k>.json` with its peak memory and status (ok / oom / error).

Usage (on a GPU node, from the repo root, with the venv active):
    python scripts/probe_frames.py --ngpu 2 --out runs/probe_frames -- <extra train.py overrides>
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback

LADDER = [21, 16, 13, 9, 5]  # latent T -> pixel frames (T - 1) * 4 + 1: 81, 61, 49, 33, 17
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------------------------------------------
def _write(out_dir: str, payload: dict) -> None:
    rank = int(os.environ.get("RANK", 0))
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
        json.dump(payload, f, indent=1)


def worker(out_dir: str) -> None:
    sys.path.insert(0, REPO)
    import torch

    import train as train_entry
    import fastgen.utils.logging_utils as logger
    from fastgen.trainer import Trainer
    from fastgen.utils.distributed import clean_up
    from fastgen.utils.scripts import parse_args, setup

    def _no_save(self, model, iteration, path=None):  # probe only: never write checkpoints
        logger.warning(f"[probe] skipping save_checkpoint at iteration {iteration}")
        return None

    Trainer.save_checkpoint = _no_save

    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_out", required=True)
    args = parse_args(parser)
    t0 = time.time()
    status, err = "ok", ""
    try:
        config = setup(args)
        train_entry.main(config)
    except torch.OutOfMemoryError as e:
        status, err = "oom", str(e).splitlines()[0]
    except Exception as e:  # NCCL errors caused by an OOM on another rank end up here too
        status, err = ("oom" if "out of memory" in str(e).lower() else "error"), traceback.format_exc()[-3000:]
    dev = torch.cuda.current_device()
    _write(
        out_dir,
        {
            "status": status,
            "error": err,
            "gpu": torch.cuda.get_device_name(dev),
            "total_gb": torch.cuda.get_device_properties(dev).total_memory / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated(dev) / 1024**3,
            "max_reserved_gb": torch.cuda.max_memory_reserved(dev) / 1024**3,
            "elapsed_s": time.time() - t0,
        },
    )
    if status == "ok":
        clean_up()
    else:
        os._exit(1)  # do not wait on collectives that will never complete


# ---------------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------------
def trial(T: int, bs: int, a: argparse.Namespace, extra: list[str]) -> dict:
    name = f"T{T}_bs{bs}"
    out_dir = os.path.abspath(os.path.join(a.out, f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"))
    os.makedirs(out_dir)
    seq_len = (T - 1) * 4 + 1
    cmd = [
        "torchrun", f"--nproc_per_node={a.ngpu}", "--standalone", os.path.abspath(__file__),
        "--probe_out", out_dir,
        "--config", a.config, "-",
        f"model.input_shape=[16,{T},60,104]",
        f"dataloader_train.sequence_length={seq_len}",
        f"dataloader_train.batch_size={bs}",
        f"trainer.max_iter={a.max_iter}",
        f"trainer.logging_iter={a.max_iter - 1}",
        "trainer.save_ckpt_iter=1000000",
        f"log_config.name=probe_{name}",
        *extra,
    ]
    env = dict(os.environ, FASTGEN_OUTPUT_ROOT=os.path.join(os.path.abspath(a.out), "fastgen_output"))
    log_path = os.path.join(out_dir, "train.log")
    print(f"[probe] {name}: seq_len={seq_len} -> {log_path}", flush=True)
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        t0, killed = time.time(), ""
        while p.poll() is None:
            time.sleep(10)
            done = [f for f in os.listdir(out_dir) if f.startswith("rank") and f.endswith(".json")]
            failed = any(json.load(open(os.path.join(out_dir, f)))["status"] != "ok" for f in done)
            if failed or time.time() - t0 > a.timeout_min * 60:
                killed = "rank failure" if failed else "timeout"
                time.sleep(20)  # give the other ranks a moment to write their own result
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
    ranks = []
    for r in range(a.ngpu):
        f = os.path.join(out_dir, f"rank{r}.json")
        ranks.append(json.load(open(f)) if os.path.exists(f) else {"status": "no_result"})
    statuses = {x["status"] for x in ranks}
    status = "ok" if statuses == {"ok"} else ("oom" if "oom" in statuses else sorted(statuses - {"ok"})[0])
    peak = max((x.get("max_allocated_gb", 0) for x in ranks), default=0)
    total = min((x.get("total_gb", 1e9) for x in ranks if "total_gb" in x), default=None)
    res = {
        "T": T, "pixel_frames": seq_len, "per_gpu_batch": bs, "status": status, "killed": killed,
        "peak_allocated_gb": peak, "peak_reserved_gb": max((x.get("max_reserved_gb", 0) for x in ranks), default=0),
        "total_gb": total, "peak_frac": (peak / total) if total else None, "ranks": ranks,
        "fits": status == "ok" and total is not None and peak <= a.max_frac * total,
    }
    print(f"[probe] {name}: status={status} peak_alloc={peak:.2f} GB "
          f"({(res['peak_frac'] or 0) * 100:.1f} %) fits={res['fits']}", flush=True)
    return res


def driver() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ngpu", type=int, default=2)
    ap.add_argument("--config", default="fastgen/configs/experiments/WanT2V/config_dmd2.py")
    ap.add_argument("--ladder", type=int, nargs="+", default=LADDER)
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 1])
    ap.add_argument("--max_iter", type=int, default=11, help="iters 1..max_iter-1 run; 5 and 10 are student steps")
    ap.add_argument("--max_frac", type=float, default=0.90)
    ap.add_argument("--timeout_min", type=float, default=60)
    ap.add_argument("--all", action="store_true", help="probe every (T, batch) instead of stopping at the first fit")
    ap.add_argument("--out", default="runs/probe_frames")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="extra train.py overrides after `--`")
    a = ap.parse_args()
    extra = [x for x in a.extra if x != "--"]
    assert all(t >= 5 for t in a.ladder), "spec §2.5: never go below T=5"
    os.makedirs(a.out, exist_ok=True)
    results, chosen = [], None
    for T in a.ladder:
        for bs in a.batch_sizes:
            r = trial(T, bs, a, extra)
            results.append(r)
            with open(os.path.join(a.out, "probe_results.json"), "w") as f:
                json.dump({"extra_overrides": extra, "results": results}, f, indent=1)
            if r["fits"] and chosen is None:
                chosen = r
                if not a.all:
                    break
        if chosen is not None and not a.all:
            break
    print("[probe] summary:")
    for r in results:
        print(f"  T={r['T']:>2} ({r['pixel_frames']:>2}f) bs={r['per_gpu_batch']} status={r['status']:<8} "
              f"peak_alloc={r['peak_allocated_gb']:.2f} GB peak_reserved={r['peak_reserved_gb']:.2f} GB "
              f"frac={(r['peak_frac'] or 0) * 100:.1f} % fits={r['fits']}")
    print(f"[probe] chosen: {None if chosen is None else (chosen['T'], chosen['per_gpu_batch'])}")


if __name__ == "__main__":
    if "--probe_out" in sys.argv:
        worker(sys.argv[sys.argv.index("--probe_out") + 1])
    else:
        driver()
