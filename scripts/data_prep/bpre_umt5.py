"""B-pre step 3.5: UMT5 caption embeddings exactly as training computes them, pooled for matching.

Uses fastgen's own `WanTextEncoder` (Wan-AI/Wan2.1-T2V-1.3B-Diffusers text_encoder + tokenizer; prompt_clean; max_length 512,
padding to max_length, truncation; padding positions zeroed), moved to cuda / bfloat16 as in training (precision
bfloat16). Matching vector = mean over the non-padding tokens (attention mask of the same tokenizer call),
l2-normalised, float32 [N, 4096]. Training itself encodes captions online, so per-token embeddings are not stored.

Usage: source .venv/bin/activate (sets HF_HOME); PYTHONPATH=$PWD python scripts/data_prep/bpre_umt5.py --captions <jsonl> \
          --field caption_uni --out runs/phaseAB/bpre/umt5/<d>_uni
"""
import argparse
import json
import os

import numpy as np
import torch

from fastgen.networks.Wan.network import WanTextEncoder
from fastgen.utils.basic_utils import prompt_clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True, nargs="+")
    ap.add_argument("--field", default="caption_uni")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")  # as fastgen/configs/net.py
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    rows = [json.loads(l) for p in a.captions for l in open(p)]
    enc = WanTextEncoder(a.model)
    enc.to(device="cuda", dtype=torch.bfloat16)
    ids, vecs, ntok = [], [], []
    for b in range(0, len(rows), a.batch):
        chunk = rows[b: b + a.batch]
        texts = [r[a.field] for r in chunk]
        with torch.no_grad():
            emb = enc.encode(texts)  # [B, 512, 4096] float32, padding zeroed
            m = enc.tokenizer([prompt_clean(t) for t in texts], padding="max_length", max_length=512,
                              truncation=True, add_special_tokens=True, return_attention_mask=True,
                              return_tensors="pt").attention_mask.to(emb.device)
            v = (emb * m[..., None]).sum(1) / m.sum(1, keepdim=True)
            v = torch.nn.functional.normalize(v.float(), dim=-1)
        ids.extend(r["clip_id"] for r in chunk)
        vecs.append(v.cpu().numpy())
        ntok.extend(m.sum(1).tolist())
        if b % (a.batch * 100) == 0:
            print(f"{b + len(chunk)}/{len(rows)}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.save(a.out + ".npy", np.concatenate(vecs).astype(np.float32))
    with open(a.out + "_ids.txt", "w") as f:
        f.write("\n".join(ids) + "\n")
    with open(a.out + "_ntok.json", "w") as f:
        json.dump(dict(zip(ids, ntok)), f)
    print("done", len(ids), "truncated(>=512 tokens):", sum(t >= 512 for t in ntok))


if __name__ == "__main__":
    main()
