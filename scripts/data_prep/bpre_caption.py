"""B-pre step 3: unified content-only recaptioning with Qwen3-VL-4B-Instruct (vLLM), same template for all datasets.

Input: normalised clips (832x480, 16 fps, 81 frames = the training window). Frames: sampled by qwen-vl-utils at
FPS_VLM (default 2 fps) from the whole clip, at the training resolution (max_pixels = 832*480).
Decoding: temperature 0.2, top-p 0.9, max_tokens 384, per-clip seed = SEED + crc32(clip_id).
Banned-word check (case-insensitive, word boundary) -> regenerate offenders once with a stricter reminder appended;
if still failing, strip the offending sentences. Writes one jsonl row per clip (resumable: existing ids are skipped).

Usage: .venv-bpre/bin/python scripts/data_prep/bpre_caption.py --index runs/phaseAB/bpre/clips/mixkit_index.jsonl \
          --clips runs/phaseAB/bpre/clips --out runs/phaseAB/bpre/captions/mixkit.jsonl [--shard i/n] [--limit N]
"""
import argparse
import json
import os
import re
import zlib

# Stated range in the template. Target output = 90-140 words (user, 2026-10-23); Qwen3-VL-4B writes shorter than
# asked: stated 90-140 -> median 82 (22 % in 90-140), 130-180 -> median 109 (98 % in 90-140) (jobs 66942-66944).
L_MIN, L_MAX = 130, 180
BANNED = ["cinematic", "realistic", "photorealistic", "CGI", "rendered", "3D", "animation", "animated", "cartoon",
          "anime", "high quality", "4K", "HD", "film", "footage", "stock", "vivid", "saturated", "beautiful",
          "stunning", "aesthetic"]
BANNED_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in BANNED) + r")\b", re.IGNORECASE)
SYSTEM_TMPL = (
    "Describe the video for a text-to-video model. Describe ONLY its content:\n"
    "the main subjects and their appearance (clothing, colour, species, objects),\n"
    "what they do and how they move, the setting and background, and the camera\n"
    "movement and framing (e.g. static, pan left, close-up, aerial).\n"
    "Do NOT describe image quality, rendering, realism or visual style. Do not use\n"
    "words such as: cinematic, realistic, photorealistic, CGI, rendered, 3D,\n"
    "animation, animated, cartoon, anime, high quality, 4K, HD, film, footage,\n"
    "stock, vivid, saturated, beautiful, stunning, aesthetic.\n"
    "Write one paragraph of {l_min}-{l_max} words."
)
USER = "Describe this video."
STRICT = ("\nReminder: your answer must not contain any of these words, in any form or capitalisation: "
          + ", ".join(BANNED) + ". Describe only what is shown, never how it looks or was made.")
SEED = 0


def banned_hits(text):
    return sorted({m.group(0).lower() for m in BANNED_RE.finditer(text)})


def strip_banned(text):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    keep = [s for s in sents if not BANNED_RE.search(s)]
    return " ".join(keep)


def clean(text):
    return " ".join(text.strip().split())


def read_video_pyav(ele):
    """qwen-vl-utils video reader backed by PyAV (bundled FFmpeg with h264). Same frame sampling and metadata as its
    torchcodec reader: torchvision here has no io.read_video and the system libavcodec cannot decode h264."""
    import av
    import numpy as np
    import torch
    import qwen_vl_utils.vision_process as vp

    path = ele["video"][len("file://"):] if ele["video"].startswith("file://") else ele["video"]
    with av.open(path) as c:
        s = c.streams.video[0]
        video_fps = float(s.average_rate)
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
    start, end, total = vp.calculate_video_frame_range(ele, len(frames), video_fps)
    nframes = vp.smart_nframes(ele, total_frames=total, video_fps=video_fps)
    idx = torch.linspace(start, end, nframes).round().long().tolist()
    video = torch.from_numpy(np.stack([frames[i] for i in idx])).permute(0, 3, 1, 2)
    meta = dict(fps=video_fps, frames_indices=idx, total_num_frames=total, video_backend="pyav")
    return video, meta, nframes / max(total, 1e-6) * video_fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="optional file with clip ids to caption (subset)")
    ap.add_argument("--fps_vlm", type=float, default=2.0)
    ap.add_argument("--attempt_offset", type=int, default=0,
                    help="seed offset for a repair pass (e.g. 2 for captions that hit max_tokens in a repetition loop)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--l_min", type=int, default=L_MIN, help="{L_MIN} in the template")
    ap.add_argument("--l_max", type=int, default=L_MAX, help="{L_MAX} in the template")
    a = ap.parse_args()

    os.environ["FORCE_QWENVL_VIDEO_READER"] = "pyav"
    import qwen_vl_utils.vision_process as vp
    from qwen_vl_utils import process_vision_info

    vp.VIDEO_READER_BACKENDS["pyav"] = read_video_pyav
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    SYSTEM = SYSTEM_TMPL.format(l_min=a.l_min, l_max=a.l_max)
    print("system prompt:\n" + SYSTEM, flush=True)
    rows = [json.loads(l) for l in open(a.index)]
    rows = [r for r in rows if r["status"] == "ok"]
    if a.ids:
        want = {l.strip() for l in open(a.ids) if l.strip()}
        rows = [r for r in rows if r["clip_id"] in want]
    si, sn = map(int, a.shard.split("/"))
    rows = rows[si::sn]
    if a.limit:
        rows = rows[: a.limit]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["clip_id"] for l in open(a.out)}
    rows = [r for r in rows if r["clip_id"] not in done]
    print(f"to caption: {len(rows)} (already done {len(done)})", flush=True)
    if not rows:
        return

    proc = AutoProcessor.from_pretrained(a.model)
    llm = LLM(model=a.model, max_model_len=8192, limit_mm_per_prompt={"image": 0, "video": 1},
              gpu_memory_utilization=0.85, seed=SEED)

    def make_input(r, strict):
        path = os.path.abspath(os.path.join(a.clips, r["dataset"], r["clip_id"] + ".mp4"))
        msgs = [{"role": "system", "content": SYSTEM + (STRICT if strict else "")},
                {"role": "user", "content": [
                    {"type": "video", "video": "file://" + path, "fps": a.fps_vlm, "max_pixels": 832 * 480},
                    {"type": "text", "text": USER}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        _, videos, vkw = process_vision_info(msgs, image_patch_size=16, return_video_kwargs=True,
                                             return_video_metadata=True)
        return {"prompt": prompt, "multi_modal_data": {"video": videos}, "mm_processor_kwargs": vkw}

    def sp(cid, attempt):
        return SamplingParams(temperature=0.2, top_p=0.9, max_tokens=384,
                              seed=(SEED + zlib.crc32(cid.encode()) + 7919 * (attempt + a.attempt_offset)) % (2**31))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a") as f:
        for b in range(0, len(rows), a.batch):
            chunk = rows[b: b + a.batch]
            outs = llm.generate([make_input(r, False) for r in chunk], [sp(r["clip_id"], 0) for r in chunk],
                                use_tqdm=False)
            res = []
            for r, o in zip(chunk, outs):
                t = clean(o.outputs[0].text)
                res.append(dict(clip_id=r["clip_id"], dataset=r["dataset"], caption_uni_raw=t,
                                hits_first=banned_hits(t), finish_first=o.outputs[0].finish_reason))
            redo = [i for i, x in enumerate(res) if x["hits_first"]]
            if redo:
                outs2 = llm.generate([make_input(chunk[i], True) for i in redo],
                                     [sp(chunk[i]["clip_id"], 1) for i in redo], use_tqdm=False)
                for i, o in zip(redo, outs2):
                    res[i]["caption_retry"] = clean(o.outputs[0].text)
                    res[i]["hits_retry"] = banned_hits(res[i]["caption_retry"])
            for x in res:
                if not x["hits_first"]:
                    x["caption_uni"], x["fix"] = x["caption_uni_raw"], "none"
                elif not x["hits_retry"]:
                    x["caption_uni"], x["fix"] = x["caption_retry"], "regenerated"
                else:
                    x["caption_uni"], x["fix"] = strip_banned(x["caption_retry"]), "stripped"
                x["hits_final"] = banned_hits(x["caption_uni"])
                x["word_count_uni"] = len(x["caption_uni"].split())
                f.write(json.dumps(x) + "\n")
            f.flush()
            print(f"{b + len(chunk)}/{len(rows)} (retry {len(redo)})", flush=True)


if __name__ == "__main__":
    main()
