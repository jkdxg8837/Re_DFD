# SPDX-License-Identifier: Apache-2.0
"""Recaption videos with Qwen2.5-VL-7B-Instruct in the style of the ViPE `long` captions (spec §2.3).

Target style (runs/NOTES_paper.md): one dense English paragraph, present tense, ~100-150 words, starting with the
main subject; appearance -> actions -> setting/lighting -> a closing mood sentence; no lists or headings.

Usage (one process per GPU; run with .venv-vlm):
  python scripts/data_prep/caption_qwen.py --list <file with one mp4 path per line> --out <out.jsonl> \
      [--shard 0 --nshards 2] [--prompt_version v1]
Output: one JSON line per video {"path", "id", "caption", "n_words", "prompt_version", "seed"}. Resumable.
"""

import argparse
import hashlib
import json
import os

import torch

PROMPTS = {
    "v1": (
        "Describe this video in one paragraph of 100 to 150 words, in English and in the present tense. "
        "Begin directly with the main subject (for example \"A young woman with long dark hair ...\"); never begin "
        "with phrases such as \"The video shows\" or \"In this video\". In this order, describe: the main subject's "
        "appearance (clothing, hair, colors, notable objects); their actions and movements as they unfold over "
        "time; the setting, background and lighting. Mention camera movement only if it is clearly noticeable. "
        "Finish with one sentence about the overall mood or atmosphere of the scene. Use concrete visual details. "
        "Write plain prose only: no lists, headings, bullet points or line breaks, and do not mention sound."
    ),
    # v2: v1 over-mentioned a static camera (82% vs 50% of the originals) and used a formulaic closing sentence
    "v2": (
        "Describe this video in one paragraph of 100 to 150 words (about five or six sentences), in English and in "
        "the present tense. Begin directly with the main subject (for example \"A young woman with long dark hair "
        "...\"); never begin with phrases such as \"The video shows\" or \"In this video\". In this order, describe: "
        "the main subject's appearance (clothing, hair, colors, notable objects); their actions and movements as "
        "they unfold over time; the setting, background and lighting. Say nothing about the camera if it is still; "
        "mention it only when it clearly moves (a pan, zoom or tracking shot). End with a sentence that conveys the "
        "mood or atmosphere, phrased naturally and not starting with \"The overall mood\" or \"The overall "
        "atmosphere\". Use concrete visual details. Write plain prose only: no lists, headings, bullet points or "
        "line breaks, and do not mention sound."
    ),
    # v3: v2's example opening ("A young woman with long dark hair ...") leaked into 44 % of Mixkit captions, often
    # describing people who are not in the clip. No example subject; describe only what is visible.
    "v3": (
        "Describe this video in one paragraph of 100 to 150 words (about five or six sentences), in English and in "
        "the present tense. Describe only what is actually visible in the frames; do not invent people, objects or "
        "events. Begin directly with the main subject or, if no person is visible, with the main object or scene; "
        "never begin with phrases such as \"The video shows\" or \"In this video\". In this order, describe: the "
        "main subject's appearance (for people: clothing, hair, colors; for objects or scenery: shape, color, "
        "texture); their actions and movements as they unfold over time; the setting, background and lighting. Say "
        "nothing about the camera if it is still; mention it only when it clearly moves (a pan, zoom or tracking "
        "shot). End with a sentence that conveys the mood or atmosphere, phrased naturally and not starting with "
        "\"The overall mood\" or \"The overall atmosphere\". Use concrete visual details. Write plain prose "
        "only: no lists, headings, bullet points or line breaks, and do not mention sound."
    ),
}
SYSTEM = "You are a helpful assistant that writes detailed, accurate descriptions of videos."
VIDEO_FPS = 2.0  # 81 frames at 16 fps = 5.06 s -> ~10 frames fed to the VLM
MAX_PIXELS = 360 * 640


def seed_of(path):
    return int(hashlib.md5(path.encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--prompt_version", default="v1")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.7)  # paper App. B.1.1 sampling
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    a = ap.parse_args()

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    paths = [l.strip() for l in open(a.list) if l.strip()][a.shard :: a.nshards]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["path"] for l in open(a.out)}
    paths = [p for p in paths if p not in done]
    print(f"shard {a.shard}/{a.nshards}: {len(done)} done, {len(paths)} to caption", flush=True)
    if not paths:
        return

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    ).eval()
    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "left"
    prompt = PROMPTS[a.prompt_version]

    with open(a.out, "a") as fo:
        for b0 in range(0, len(paths), a.batch):
            batch = paths[b0 : b0 + a.batch]
            msgs = [
                [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": os.path.abspath(p), "fps": VIDEO_FPS, "max_pixels": MAX_PIXELS},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                for p in batch
            ]
            texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            _, videos, vkw = process_vision_info(msgs, return_video_kwargs=True)
            inputs = proc(text=texts, videos=videos, padding=True, return_tensors="pt", **vkw).to("cuda")
            # one seed per batch, derived from its first path, so reruns reproduce the captions
            torch.manual_seed(seed_of(batch[0]))
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=a.temperature,
                    top_p=a.top_p,
                    max_new_tokens=a.max_new_tokens,
                )
            caps = proc.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            for p, c in zip(batch, caps):
                c = " ".join(c.split())
                rec = {"path": p, "id": os.path.basename(p)[:-4], "caption": c, "n_words": len(c.split()),
                       "prompt_version": a.prompt_version, "seed": seed_of(batch[0])}
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            print(f"{b0 + len(batch)}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
