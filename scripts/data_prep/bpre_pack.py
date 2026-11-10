"""B-pre step 7: WDS shards in the format of the E2 Mixkit shard (`data/mixkit_wds_v3`): per sample `<key>.mp4` +
`<key>.txt`, key = %08d, 100 samples per shard (`%06d.tar`), sample order = seed-0 shuffle; no stored embeddings
(training encodes the caption online). mp4 = the normalised clip (832x480, 16 fps, 81 frames).

  pack  : --dataset D --subset sub1 --caption uni|orig -> <out>/<D>_<subset>_<caption>/ (+ manifest.jsonl)
  check : load 4 samples through the training dataloader (VideoWDSLoader, E1a/E2 arguments), print shapes and
          captions, save a contact sheet; the same for the E2 shard as the shape reference. (training venv)
"""
import argparse
import io
import json
import os
import random
import tarfile

import pandas as pd


def cmd_pack(a):
    df = pd.read_parquet(f"{a.root}/index.parquet").set_index("clip_id")
    ids = [l.strip() for l in open(f"{a.root}/splits/{a.dataset}_{a.subset}.txt") if l.strip()]
    random.Random(0).shuffle(ids)
    out = os.path.join(a.out, f"{a.dataset}_{a.subset}_{a.caption}")
    assert not os.path.exists(out), f"{out} exists; never overwrite a shard"
    os.makedirs(out)
    col = "caption_uni" if a.caption == "uni" else "caption_orig"
    man = open(os.path.join(out, "manifest.jsonl"), "w")
    tf = None
    for i, cid in enumerate(ids):
        if i % 100 == 0:
            if tf:
                tf.close()
            shard = f"{i // 100:06d}.tar"
            tf = tarfile.open(os.path.join(out, shard), "w")
        key = f"{i:08d}"
        cap = df.loc[cid, col].strip()
        for ext, data in [("mp4", open(os.path.join(a.root, "clips", a.dataset, cid + ".mp4"), "rb").read()),
                          ("txt", cap.encode("utf-8"))]:
            ti = tarfile.TarInfo(f"{key}.{ext}")
            ti.size, ti.mode = len(data), 0o644
            tf.addfile(ti, io.BytesIO(data))
        man.write(json.dumps(dict(key=key, shard=shard, clip_id=cid)) + "\n")
    tf.close()
    man.close()
    print(out, len(ids), "samples", (len(ids) + 99) // 100, "shards")


def cmd_check(a):
    import numpy as np
    import torch
    from PIL import Image

    from fastgen.datasets.wds_dataloaders import VideoWDSLoader

    def load4(path):
        dl = VideoWDSLoader(datatags=[f"WDS:{path}"], batch_size=1, key_map={"real": "mp4", "condition": "txt"},
                            presets_map={"neg_condition": "neg_prompt_wan"}, sequence_length=81, img_size=(832, 480),
                            decoder_type="full", num_workers=1, shuffle_size=100)
        it = iter(dl)
        return [next(it) for _ in range(4)]

    ref = load4(a.ref_shard)[0]
    ref_shapes = {k: tuple(v.shape) for k, v in ref.items() if torch.is_tensor(v)}
    print("E2 reference shard shapes:", ref_shapes)
    os.makedirs(a.sheet_dir, exist_ok=True)
    for path in a.shards:
        name = os.path.basename(path.rstrip("/"))
        batch = load4(path)
        tiles = []
        for b in batch:
            shapes = {k: tuple(v.shape) for k, v in b.items() if torch.is_tensor(v)}
            assert shapes == ref_shapes, (shapes, ref_shapes)
            v = ((b["real"][0].float() + 1) * 127.5).clamp(0, 255).byte()  # c t h w
            tiles.append(torch.cat([v[:, t] for t in (0, 40, 80)], dim=2).permute(1, 2, 0).numpy())
            print(f"[{name}] shapes {shapes} | caption: {b['condition'][0][:300]}")
        sheet = np.concatenate(tiles, axis=0)
        Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(f"{a.sheet_dir}/{name}.jpg")
        print(f"[{name}] OK: same tensor shapes as the E2 shard; contact sheet {a.sheet_dir}/{name}.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pack", "check"])
    ap.add_argument("--root", default="runs/phaseAB/bpre")
    ap.add_argument("--out", default="runs/phaseAB/bpre/shards")
    ap.add_argument("--dataset")
    ap.add_argument("--subset")
    ap.add_argument("--caption", choices=["uni", "orig"])
    ap.add_argument("--shards", nargs="*", default=[])
    ap.add_argument("--ref_shard", default="data/mixkit_wds_v3")
    ap.add_argument("--sheet_dir", default="runs/phaseAB/bpre/shard_check")
    a = ap.parse_args()
    {"pack": cmd_pack, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
