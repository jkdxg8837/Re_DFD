# Real-data pipeline for E2 / E2s (Mixkit), spec §2.3

Decisions (spec, all [CONFIRMED 2026-09-28]): source HF `FastVideo/Mixkit-Src`; scene-cut filtering with
PySceneDetect only; recaption with Qwen2.5-VL-7B-Instruct matched to the HF ViPE `long` style (~100–150 words);
at most one clip per source video; portrait videos dropped; main dataset deferred.

## Steps (scripts in `scripts/data_prep/`, env `.venv-vlm`: torch 2.8.0+cu126, transformers 4.57.1, qwen-vl-utils 0.0.14, scenedetect 0.7.1)
1. Download `FastVideo/Mixkit-Src` (8,357 pre-cut clips in 57 category folders, 1920×1080 (3.5 % portrait),
   30 fps, 6 s = 180 frames) to `data/mixkit_src/`. Anonymous download hit HF 429 rate limits; resumed with the
   user's HF token via `HF_TOKEN_PATH` (not copied).
2. `prep_mixkit.py` → `data/mixkit_prep/{clips/,meta.jsonl}`: one clip per source id (lowest `_clip_N`); drop
   portrait; decode at 1280×720 (area filter); 16 fps by nearest-frame selection (src frame round(k·30/16));
   middle 81 frames (96 available at 16 fps); drop if PySceneDetect ContentDetector (threshold 27, min_scene_len 15)
   finds a cut inside the 81 frames; libx264 crf 18 yuv420p.
   - Stored like ViPE `long` (h264 1280×720, 16 fps, 81 frames, yuv420p); the loader resizes the small side to 480
     and center-crops 832×480 online, identical for both datasets. With `decoder_type=full` the loader uses frames
     [0, 81) of the stored clip, so storing exactly the chosen 81 frames fixes the window.
   - Difference: bitrate. Mixkit clips ≈ 4.6 MB (crf 18) vs ViPE ≈ 0.7–1.2 MB (≈ 1 Mbps). Re-encoding noise is
     part of "real vs synthetic"; noted, not matched.
   - Test on 40 sources: 36 ok, 3 portrait, 1 scene cut.
3. `caption_qwen.py` (prompt v2, T=0.7, top-p=0.8 as in the paper's App. B.1.1, 2 fps / ≤ 360×640 px per frame,
   max 320 new tokens; seed per batch from the path).
4. `build_wds.py` → `data/mixkit_wds/` (E2: mp4 + txt) and `data/mixkit_vipe_pair_wds/` (E2s: + `student.mp4`, the
   ViPE `long` video whose caption is the all-mpnet-base-v2 nearest neighbor of the Mixkit caption, with
   replacement; + `pair.json`). Same seeded shuffle, 100 samples per shard.

## Prompt validation (50 random ViPE `long` videos, `data/caption_val/`, reports `caption_val_v{1,2}.md`)

| stat | v1 | v2 (chosen) | original |
|---|---|---|---|
| words mean (p5–p95) | 127.8 (90–153) | 116.4 (90–143) | 122.7 (96–146) |
| sentences | 6.8 | 6.2 | 5.7 |
| mentions "camera" | 82 % | 34 % | 50 % |
| starts with "A"/"An" | 90 % | 90 % | 74 % |
| matched-pair cosine (mpnet) / mismatched | 0.846 / 0.355 | 0.851 / 0.336 | – |
| retrieval top-1 (Qwen → original) | 0.94 | 0.98 | – |

v2 changes: "about five or six sentences", no camera sentence when the camera is still, no "The overall mood…"
closing (Qwen still sometimes writes "The overall atmosphere is one of…").

## 2026-09-28: v2 captions rejected (prompt-example leakage)

Full Mixkit run with v2 (5,227 clips, 2 GPUs, 53 min; ~1.25 s/clip/GPU). Caption vs Mixkit title check
(`check_caption_title.py`, mpnet cosine between caption and the human-written filename title): mean 0.50,
13.6 % below the random-pair p95. The lowest clips showed **44.3 % of all captions start with "A young woman with long
dark hair"** — the example opening in the v2 prompt — including clips with no person (empty basketball court, sky,
raindrops, particles) or with men only (contact sheet `caption_title_check_lowest.jpg`). The ViPE validation also had
11/50 such openings but its statistics (length, retrieval 0.98) did not reveal it. The v2 WDS
(`data/mixkit_wds`, `data/mixkit_vipe_pair_wds`) is kept (never delete data) but **not used**; SMOKE_E2 on it was
cancelled after 1 min. Fix: prompt v3 (no example subject, "describe only what is visible"); validation now also
reports the most common 5-word opening and the caption–title check on 100 random Mixkit clips.
Also seen: some low-cosine clips are fine captions of an off-title window (the middle 81 frames of a 6 s clip, e.g.
"boys playing basketball" whose window shows only the empty court) — the title check over-flags those.

## Prompt v3 validation (job 2222224)

| stat | v2 | v3 | original ViPE |
|---|---|---|---|
| ViPE-50: words mean (p5–p95) | 116.4 (90–143) | 116.3 (89–147) | 122.7 (96–146) |
| ViPE-50: opens "A young woman with long dark hair" | 22 % | 0 % | 0 % |
| ViPE-50: most common 5-word opening | 22 % | 4 % | 4 % |
| ViPE-50: mentions camera | 34 % | 22 % | 50 % |
| ViPE-50: starts "The video …/In this …" | 4 % | 10 % | 4 % |
| ViPE-50: matched cos / retrieval top-1 | 0.851 / 0.98 | 0.861 / 1.00 | – |
| Mixkit-100: opens "A young woman with long dark hair" | 47 | 0 | – |
| Mixkit-100: caption–title cos mean | 0.489 | 0.498 | – |

The 12 lowest-title-similarity Mixkit clips under v3 (`caption_title_mixkit100_v3_lowest.jpg`) were checked by
eye: all captions match the frames (the title describes the full 6 s clip or intent, not the middle 81 frames).
Full v3 run → `data/mixkit_prep/captions_v3.jsonl`, WDS `data/mixkit_wds_v3`, `data/mixkit_vipe_pair_wds_v3`
(used by E2 / E2s, `runs/sbatch/e2_common.sh`).

## E2s memory probe (job 2222223, runs/probe_frames_E2s)
T=21, per-GPU batch 1, E2s + F2 diagnostics: ok, peak allocated 30.19 GB (76.4 % of 39.5 GB; E1a 29.98 GB).
First attempt failed on a config error (`key_map={...}` override not allowed in struct mode), fixed with
`++dataloader_train.key_map.student_real=student.mp4`.
- 2026-09-28: SMOKE_E1d / SMOKE_E2 / SMOKE_E2s checkpoints (3 × 32 GB) deleted at the user's request; logs, config, load_report, wandb kept.
- 2026-09-28: SMOKE_E3 / SMOKE_E3s checkpoints deleted (standing user rule: delete smoke checkpoints once the smoke test passes).
