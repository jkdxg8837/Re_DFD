# Paper notes: DFD, arXiv 2606.18478v1 (16 Jun 2026)

Read on 2026-09-27 from the arXiv HTML version (text extracted locally, quotes verbatim). Question asked by the user:
what does the paper say about critic (fake score) initialization?

## Critic / fake-score initialization

**The paper does not state how the fake score (critic) is initialized for DFD post-training, and never mentions a
critic warm-up.** Relevant passages:

- §3 (background): the fake score "is learned by an auxiliary denoiser trained online on student samples via
  denoising score matching … Training alternates between updating [the student] … and updating the auxiliary
  denoiser to track the moving [student distribution]."
- §4 (validity condition): "we satisfy it implicitly by applying DFD as a post-training stage on top of a
  DMD2-pretrained model, whose generations are already close to the real video".
- §5.2 "Effects of DMD2 Pretraining": "we ablate by initializing the **student** with the teacher model (without DMD2
  distillation) and comparing the DMD2 initialization … Even after a long training run (e.g., 1400 iterations), the
  model fails to converge". This is about the student's init, not the critic's.
- Algorithm 1 only shows the student update step (`fake_score_network` is used, not initialized).
- Table 4 (training configurations), Wan2.1 T2V DFD column: "Pretrained Model Iterations: 25k (from DMD2)". It
  does not say which sub-networks are taken from that checkpoint.

So "the authors continued with the DMD2-trained critic" is still an inference, from (i) the README command loading
`{net, fake_score, discriminator}` from the DMD2 checkpoint and (ii) the `.pth` scheduler states
(`runs/PHASE0.md` §3). It is not stated in the paper.

## Other facts from the paper that matter for E1

| item | paper (Wan2.1 T2V) | released checkpoints / our E1a |
|---|---|---|
| DMD2 pretraining before DFD | **25k iterations** (Table 4) | `dmd2_t2v_vipe/0001000.pth`: iteration **1000** (200 gen + 800 critic updates) |
| DFD post-train length | **100 iterations** (Table 4); text: "50-100 steps" / "100–300 steps" | `ours_wan_t2v_vipe/0000040`: 40 iterations; E1a: 40 |
| GAN in DFD | "we drop the GAN loss completely" (§4); discriminator LR "N/A" for DFD | consistent with the `.pth` (no discriminator scheduler) and our `gan_loss_weight_gen=0` |
| CFG scale | 5.0 | 5.0 |
| student update frequency | 5 | 5 |
| batch / frames / res. | 16 / 81 / 480p | 16 / 81 / 480p |
| hardware | "8 A100 GPUs" for T2V (memory size not stated) → per-GPU batch 2 | 2× A100-40GB, per-GPU batch 1 × accum 8 |
| lr (student / fake score) | 1e-5 / 1e-5 | 1e-5 / 1e-5 |
| data | ViPE-Wild-1M (Wan2.1-14B generated), clustered into 1,000 clusters, clusters/videos **manually selected**; experiments on an "animation set" and a "mix-style set" (§4.4, App. B.1) | HF `vipe_fliter_complete/long` (29,057 clips); relation to the paper's curated sets unknown |
| diversity metric | DP-DMD metric: pairwise cosine similarity of mean-pooled **DINOv2-ViT-L** and **CLIP-ViT-L** frame features over several seeds per prompt (App. B.3.1) | ours: DINOv2-ViT-B/14, 9 frames, 5 seeds, 1 − cos |

Discrepancies worth noting:
1. The released DMD2 start (1000 iterations) is not the paper's 25k-iteration DMD2 model, and the released DFD result
   (40 iterations) is not the paper's 100-iteration setting. The released pair is internally consistent (the `ours`
   `.pth` shows 8 + 32 updates on top of a start, consistent with starting from `dmd2 … 0001000`), but its numbers
   need not match the paper's tables.
2. The paper's data is a manually curated subset of ViPE; the released `long` folder may differ from it.
3. The paper measures diversity with DINOv2-L and CLIP-L. Our DINOv2-B metric is similar in spirit but not identical.

## Caption / prompt details (checked 2026-09-28, for aligning our real-data recaptioning)

Paper App. B.1.1 "Tri-Level Captioning" (verbatim numbers recovered from the MathML):
- VLM **Qwen3-VL-4B-Instruct** (vLLM), one forward pass per video, one system prompt asking for tagged output:
  `[LONG]` **250–300 words** (subject appearance, actions, expressions, environment, camera work, visual style),
  `[MEDIUM]` **150–180 words** (key subject, actions, setting, style), `[SHORT]` one sentence.
- Sampling: **T=0.7, top-p=0.8, max_new_tokens=512**. Tags are parsed by regex. Fallback: truncate LONG for MEDIUM,
  first sentence for SHORT. Captions are also encoded with UMT5-XXL (Wan2.1-T2V-1.3B).
- **The system prompt text itself is not published** (not in the paper, repo, or HF dataset card;
  `csusupergear/vipe_fliter_complete` has no README; `nvidia/vipe-wild-sdg-1m/meta.parquet` has only
  `sequence, num_frames, tar_name`, no captions or prompts).

Measured on what we actually train with (`data/vipe_fliter_complete/long`, 1,500 captions from 15 random shards):
**mean 120 words, median 121, p5–p95 93–148, max 182, ≈ 5.6 sentences**; **0** captions in the paper's LONG range
(250–300), 62 in the MEDIUM range (150–180). Style: one dense descriptive English paragraph, present tense, scene /
subject appearance → actions → surroundings/lighting → closing mood sentence; no tags, no lists, little explicit
camera vocabulary. **So the HF `long` captions do not follow the paper's tri-level spec. Their origin is unknown**
(possibly the prompts used to generate the ViPE-Wild-SDG videos with Wan2.1-14B).

Training-time text conditioning (both DMD2 and DFD, from code): positive = the sample's `.txt` caption; negative =
preset `neg_prompt_wan` (same text as `scripts/inference/prompts/negative_prompt.txt`), used for the teacher CFG
(w=5) unconditional branch.
