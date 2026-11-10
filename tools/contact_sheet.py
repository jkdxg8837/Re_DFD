# SPDX-License-Identifier: Apache-2.0
"""4x4 contact sheet (middle frame) of the first 16 videos of a generated set (task2.md A1).
Usage: python tools/contact_sheet.py <set> <out.jpg>   (reads runs/phaseAB/latents/<set>/meta.jsonl)"""

import json
import sys

import av
from PIL import Image, ImageDraw

name, out = sys.argv[1], sys.argv[2]
meta = [json.loads(l) for l in open(f"runs/phaseAB/latents/{name}/meta.jsonl")]
# spread the 16 over the prompt sources: every 200/16-th prompt
pick = [meta[int(i * len(meta) / 16)] for i in range(16)]
W, H = 416, 240
sheet = Image.new("RGB", (4 * W, 4 * (H + 18)), "white")
d = ImageDraw.Draw(sheet)
for k, m in enumerate(pick):
    with av.open(m["mp4"]) as c:
        fr = [f.to_image() for f in c.decode(video=0)]
    im = fr[len(fr) // 2].resize((W, H))
    x, y = (k % 4) * W, (k // 4) * (H + 18)
    sheet.paste(im, (x, y + 18))
    d.text((x + 4, y + 3), f"p{m['idx']:03d} [{m['source']}] {m['prompt'][:48]}", fill="black")
sheet.save(out, quality=88)
print(out)
