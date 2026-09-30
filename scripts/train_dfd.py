# SPDX-License-Identifier: Apache-2.0
"""`train.py` plus a per-submodule load report (spec §0 rule 3, Claude.md hard rule).

Same CLI and training behavior as `train.py`: it calls `train.main` unchanged. The only addition is a wrapper
around `Trainer.load_pretrained_ckpt`: after the original loading runs, it reports for each of `net`, `fake_score`,
`discriminator` and `teacher` the parameter count, dtypes, where the weights came from, and the missing / unexpected
keys against that source:
  * checkpoint key (from `pretrained_ckpt_key_map`)       -> DCP metadata `<prefix>.<key>_model/.metadata`
  * `net` / `teacher` / `fake_score` not in the key map   -> HF Wan transformer safetensors index (`transformer.` prefix)
  * `discriminator` not in the key map                    -> random init (no source)
The report is logged on rank 0 and written to `<log_config.save_path>/load_report.json`. Any non-empty missing
list is logged at CRITICAL level.

Usage: torchrun ... scripts/train_dfd.py --config <cfg> - <overrides>   (identical to train.py)
"""

import glob
import json
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import argparse  # noqa: E402

import train as train_entry  # noqa: E402
import fastgen.utils.logging_utils as logger  # noqa: E402
from fastgen.trainer import Trainer  # noqa: E402
from fastgen.utils.distributed import clean_up, is_rank0  # noqa: E402
from fastgen.utils.scripts import parse_args, setup  # noqa: E402

SUBMODULES = ["net", "fake_score", "discriminator", "teacher"]


def _hf_transformer_keys(model_id: str) -> tuple[str, set]:
    repo = "models--" + model_id.replace("/", "--")
    hits = glob.glob(os.path.join(os.environ["HF_HOME"], repo, "snapshots", "*", "transformer",
                                  "diffusion_pytorch_model.safetensors.index.json"))
    if not hits:
        return f"HF {model_id} (index not found)", None
    return f"HF {model_id}", {"transformer." + k for k in json.load(open(hits[0]))["weight_map"]}


def _dcp_keys(path: str) -> set:
    from torch.distributed.checkpoint import FileSystemReader

    return set(FileSystemReader(path).read_metadata().state_dict_metadata.keys())


def load_report(trainer: Trainer, model) -> dict:
    cfg = trainer.config.trainer.checkpointer
    key_map = dict(cfg.pretrained_ckpt_key_map or {})
    report = {}
    for name in SUBMODULES:
        m = getattr(model, name, None)
        if m is None:
            report[name] = {"present": False}
            continue
        keys = set(m.state_dict().keys())
        entry = {
            "present": True,
            "n_params": int(sum(p.numel() for p in m.parameters())),
            "dtypes": dict(Counter(str(p.dtype) for p in m.parameters())),
            "n_state_dict_keys": len(keys),
        }
        if name in key_map and cfg.pretrained_ckpt_path:
            src = f"{cfg.pretrained_ckpt_path}.{key_map[name]}_model"
            ref = _dcp_keys(src)
        elif name == "discriminator":
            src, ref = "random init", None
        else:
            src, ref = _hf_transformer_keys(getattr(m, "model_id", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"))
        entry["source"] = src
        if ref is not None:
            entry["missing"] = sorted(keys - ref)  # in the module, absent from the source
            entry["unexpected"] = sorted(ref - keys)  # in the source, absent from the module
        report[name] = entry
    return report


_orig_load = Trainer.load_pretrained_ckpt


def _load_with_report(self, model, *args, **kwargs):
    out = _orig_load(self, model, *args, **kwargs)
    if is_rank0():
        rep = load_report(self, model)
        for name, e in rep.items():
            if not e["present"]:
                logger.info(f"[load-report] {name}: not present")
                continue
            miss, unexp = e.get("missing"), e.get("unexpected")
            msg = (f"[load-report] {name}: source={e['source']} params={e['n_params'] / 1e9:.4f}B dtypes={e['dtypes']} "
                   f"missing={None if miss is None else len(miss)} unexpected={None if unexp is None else len(unexp)}")
            (logger.critical if miss else logger.info)(msg + (f" first missing: {miss[:5]}" if miss else ""))
            if unexp:
                logger.warning(f"[load-report] {name}: first unexpected: {unexp[:5]}")
        path = os.path.join(self.config.log_config.save_path, "load_report.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(rep, open(path, "w"), indent=1)
        logger.info(f"[load-report] written to {path}")
    return out


Trainer.load_pretrained_ckpt = _load_with_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training (train.py + load report)")
    args = parse_args(parser)
    config = setup(args)
    train_entry.main(config)
    clean_up()
    logger.info("Training finished.")
