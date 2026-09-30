# SPDX-License-Identifier: Apache-2.0
"""Side-by-side comparison videos from the VBench-subset eval videos (same prompt, same seed = same initial noise).

For each selected prompt writes
  strips/<nn>_<prompt>.mp4  models left to right at full resolution (832x480 each), seed --seed
  seeds/<nn>_<prompt>.mp4   models as columns, seeds 0-4 as rows, half resolution (shows seed diversity)
plus all_strips.mp4 (strips concatenated) and overview.jpg (middle frame of every strip).

Usage: python scripts/eval/splice_compare.py --out runs/compare/E0_E1a_E2 \
         --model E0_dmd2_T21 "E0 | DMD2 start (no DFD)" --model E1a_seed1_it0040 "E1a | DFD + ViPE, step 40" ...
         [--n_prompts 12] [--select_seed 0] [--seed 0]
Streams frames (low memory, no torch); runs on the login node.
"""

import argparse
import os
import random
import re
import textwrap

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
FPS = 16


def frames(path, w, h):
    with av.open(path) as c:
        for f in c.decode(video=0):
            yield f.to_image().resize((w, h), Image.BILINEAR) if (f.width, f.height) != (w, h) else f.to_image()


def text_bar(width, height, lines, font, color="white", bg=(20, 20, 20)):
    im = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(im)
    y = 6
    for ln in lines:
        d.text((10, y), ln, font=font, fill=color)
        y += font.size + 4
    return im


def write_video(out, frame_iter, size):
    tmp = out + ".tmp.mp4"
    with av.open(tmp, "w") as oc:
        s = oc.add_stream("libx264", rate=FPS, options={"crf": "18", "preset": "fast", "threads": "2"})
        s.width, s.height, s.pix_fmt = size[0], size[1], "yuv420p"
        n = 0
        for im in frame_iter:
            for pkt in s.encode(av.VideoFrame.from_image(im)):
                oc.mux(pkt)
            n += 1
        for pkt in s.encode():
            oc.mux(pkt)
    os.replace(tmp, out)
    return n


def compose(models, paths, prompt, w, h, cols, rows, label_font, prompt_font, row_labels=None):
    """Yield composed frames: header (model labels), grid of rows x cols videos, footer (prompt)."""
    n_lines = max(len(label.split("\n")) for _, label in models)  # a label may span several lines ("\n")
    head_h = (label_font.size + 6) * n_lines + 10
    wrap = textwrap.wrap(f"Prompt: {prompt}", width=max(40, cols * w // (prompt_font.size * 0.55)))
    foot_h = (prompt_font.size + 4) * len(wrap) + 12
    foot_h += (head_h + rows * h + foot_h) % 2  # yuv420p needs even dimensions
    W, H = cols * w, head_h + rows * h + foot_h
    head = Image.new("RGB", (W, head_h), (20, 20, 20))
    d = ImageDraw.Draw(head)
    for j, (_, label) in enumerate(models):
        for li, ln in enumerate(label.split("\n")):
            d.text((j * w + 10, 6 + li * (label_font.size + 6)), ln, font=label_font, fill="white" if li == 0 else (200, 220, 255))
    foot = text_bar(W, foot_h, wrap, prompt_font, color=(230, 230, 230))
    its = [[frames(paths[r][j], w, h) for j in range(cols)] for r in range(rows)]

    def gen():
        while True:
            canvas = Image.new("RGB", (W, H), "black")
            canvas.paste(head, (0, 0))
            try:
                for r in range(rows):
                    for j in range(cols):
                        canvas.paste(next(its[r][j]), (j * w, head_h + r * h))
            except StopIteration:
                return
            if row_labels:
                dd = ImageDraw.Draw(canvas)
                for r, rl in enumerate(row_labels):
                    dd.text((6, head_h + r * h + 4), rl, font=prompt_font, fill="white", stroke_width=2, stroke_fill="black")
            canvas.paste(foot, (0, head_h + rows * h))
            yield canvas

    return gen(), (W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", nargs=2, action="append", metavar=("TAG", "LABEL"), required=True)
    ap.add_argument("--prompts", default="runs/eval/vbench_subset/prompts.txt")
    ap.add_argument("--n_prompts", type=int, default=12)
    ap.add_argument("--select_seed", type=int, default=0, help="RNG seed for choosing the prompts")
    ap.add_argument("--seed", type=int, default=0, help="which of the 5 sample seeds the strips show")
    a = ap.parse_args()
    prompts = [l.rstrip("\n") for l in open(a.prompts) if l.strip()]
    sel = sorted(random.Random(a.select_seed).sample(range(len(prompts)), a.n_prompts))
    for sub in ("strips", "seeds"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    lf, pf = ImageFont.truetype(FONT_B, 22), ImageFont.truetype(FONT, 18)
    lf_s, pf_s = ImageFont.truetype(FONT_B, 16), ImageFont.truetype(FONT, 15)
    vid = lambda tag, p, k: os.path.join("runs/eval", tag, "videos", f"{p}-{k}.mp4")
    strips, mids = [], []
    with open(os.path.join(a.out, "prompts.txt"), "w") as fp:
        for n, i in enumerate(sel):
            p = prompts[i]
            fp.write(f"{n:02d}\t{i}\t{p}\n")
            name = f"{n:02d}_" + re.sub(r"[^A-Za-z0-9]+", "_", p)[:60].strip("_")
            # full-resolution strip, one seed
            it, size = compose(a.model, [[vid(t, p, a.seed) for t, _ in a.model]], p, 832, 480, len(a.model), 1, lf, pf)
            out = os.path.join(a.out, "strips", name + ".mp4")
            buf = []

            def keep(g):
                for k, im in enumerate(g):
                    if k == 40:
                        buf.append(im.copy())
                    yield im

            write_video(out, keep(it), size)
            strips.append(out)
            mids.append(buf[0])
            # seeds grid, half resolution
            paths = [[vid(t, p, k) for t, _ in a.model] for k in range(5)]
            it, size = compose(a.model, paths, p, 416, 240, len(a.model), 5, lf_s, pf_s,
                               row_labels=[f"seed {k}" for k in range(5)])
            write_video(os.path.join(a.out, "seeds", name + ".mp4"), it, size)
            print(f"{n:02d} {p}", flush=True)

    # all strips back to back (same frame size)
    def all_frames():
        for s in strips:
            yield from frames(s, *mids[0].size)

    write_video(os.path.join(a.out, "all_strips.mp4"), all_frames(), mids[0].size)
    # overview: middle frame of every strip, scaled to half
    w, h = mids[0].size[0] // 2, mids[0].size[1] // 2
    ov = Image.new("RGB", (w, h * len(mids)), "black")
    for k, m in enumerate(mids):
        ov.paste(m.resize((w, h)), (0, k * h))
    ov.save(os.path.join(a.out, "overview.jpg"), quality=85)
    print("DONE", a.out)


if __name__ == "__main__":
    main()
