# SPDX-License-Identifier: Apache-2.0
"""Caption sanity check for Mixkit: cosine between each Qwen caption and the clip's human-written Mixkit title
(the filename, e.g. "mixkit-a-bearded-man-crossing-his-arms-and-traffic-on-a-2529" -> "a bearded man crossing his
arms and traffic on a"). Low similarity flags possible VLM hallucinations or unusual footage.

Usage: python scripts/data_prep/check_caption_title.py <captions.jsonl> [--out runs/data_prep/caption_title_check]
Writes <out>.json (per-clip cosine, sorted), <out>.md (distribution + lowest 20) and a contact sheet of frames
(first/middle/last) of the lowest 12 clips to <out>_lowest.jpg. Run with .venv-vlm on a GPU node.
"""

import argparse
import json
import re

import numpy as np


def title_of(clip_id):
    t = re.sub(r"^mixkit-", "", clip_id)
    t = re.sub(r"-\d+$", "", t)
    return t.replace("-", " ")


def contact_sheet(paths, out, cols=3):
    import av
    from PIL import Image, ImageDraw

    tiles = []
    for p in paths:
        with av.open(p) as c:
            fr = [f.to_image() for i, f in enumerate(c.decode(video=0)) if i in (0, 40, 80)]
        w, h = 320, 180
        row = Image.new("RGB", (w * 3, h + 16), "white")
        for i, f in enumerate(fr):
            row.paste(f.resize((w, h)), (i * w, 16))
        ImageDraw.Draw(row).text((4, 2), p.split("/")[-1][:90], fill="black")
        tiles.append(row)
    sheet = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)), "white")
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height
    sheet.save(out, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captions")
    ap.add_argument("--out", default="runs/data_prep/caption_title_check")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.captions)]
    titles = [title_of(r["id"]) for r in recs]

    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    ec = m.encode([r["caption"] for r in recs], batch_size=256, normalize_embeddings=True)
    et = m.encode(titles, batch_size=256, normalize_embeddings=True)
    cos = (ec * et).sum(1)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(recs))
    cos_rand = (ec * et[perm]).sum(1)
    order = np.argsort(cos)
    res = [{"id": recs[i]["id"], "title": titles[i], "cos": float(cos[i]), "caption": recs[i]["caption"],
            "path": recs[i]["path"]} for i in order]
    json.dump(res, open(a.out + ".json", "w"), indent=1, ensure_ascii=False)
    q = lambda x, p: float(np.percentile(x, p))
    lines = [f"# Caption vs Mixkit title (all-mpnet-base-v2), {len(recs)} clips", "",
             f"matched cos: mean {cos.mean():.3f}, p1 {q(cos, 1):.3f}, p5 {q(cos, 5):.3f}, p10 {q(cos, 10):.3f}, "
             f"median {q(cos, 50):.3f}", f"random title: mean {cos_rand.mean():.3f}, p95 {q(cos_rand, 95):.3f}, "
             f"p99 {q(cos_rand, 99):.3f}", "",
             f"fraction below the random-pair p95 ({q(cos_rand, 95):.3f}): {float(np.mean(cos < q(cos_rand, 95))):.3f}",
             f"fraction below the random-pair p99 ({q(cos_rand, 99):.3f}): {float(np.mean(cos < q(cos_rand, 99))):.3f}",
             "", "## Lowest 20", "", "| cos | title | caption (start) |", "|---|---|---|"]
    for r in res[:20]:
        lines.append(f"| {r['cos']:.3f} | {r['title']} | {r['caption'][:140]} |")
    open(a.out + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    contact_sheet([r["path"] for r in res[:12]], a.out + "_lowest.jpg")


if __name__ == "__main__":
    main()
