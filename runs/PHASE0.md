# Phase 0 findings

Date: 2026-09-23. Cluster: Sulis (login03; GPU nodes gpu[001-030] = 3× A100-PCIE-40GB, driver 560.28.03 / CUDA 12.6).
Details of the code paths referenced below: `runs/NOTES_code_reading.md`.

## 1. Repo and environment

- Repo: `csy2077/data-forcing-distillation`, commit **`de6416c`** ("Revise citation for Data-Forcing Distillation
  paper"), working tree clean apart from the experiment files. It is checked out at
  `/gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation` (not renamed to `FastGen`; nothing depends on the
  directory name).
- venv: `.venv/` (Python 3.12.7), `pip install -e .`.
  - `torch==2.8.0+cu126`, `torchvision==0.23.0+cu126` (the cu126 wheels match the 560 driver; newer default
    PyPI torch builds need a newer driver). `diffusers==0.35.1`, `transformers==4.49.0`, `numpy==1.26.4`.
  - Missing from `requirements.txt` but imported unconditionally: **`pandas`** (`fastgen/callbacks/gpu_stats.py`),
    so it was installed. For tests: `pytest`, `setuptools<81` (`pkg_resources`). The optional imports
    `flash_attn`, `xformers`, `decord`, `pyspng` and `qwen_vl_utils` are guarded or not used by Wan T2V.
  - `source .venv/bin/activate` also exports `HF_HOME=<repo>/FASTGEN_OUTPUT/.cache` and `FASTGEN_OUTPUT_ROOT`.
- Teacher weights: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` is pre-downloaded into `HF_HOME` (28.9 GB).
  On an A100 node, the tokenizer, VAE (126.9 M), transformer (1.419 B), UniPC scheduler and UMT5 text encoder
  all load from this cache, and the cache is not modified.
  Note: diffusers 0.35.1 calls the Hub API when loading the sharded transformer even with `HF_HUB_OFFLINE=1`, so
  offline mode fails. Compute nodes have internet, so leave it unset.

## 2. Checkpoint layout

Downloaded `csusupergear/wan_t2v_checkpoints_release` to `checkpoints/wan_t2v_checkpoints_release/` and
unzipped the DCP archives in place (the zips are kept):

```
checkpoints/wan_t2v_checkpoints_release/
├── dmd2_t2v_vipe/
│   ├── 0001000.pth              # 1.57 kB: scheduler / grad_scaler / callbacks / iteration
│   ├── 0001000.net_model/       # DCP dir: .metadata + 16 × __*_0.distcp (2.7 GB)
│   └── 0001000.net_model.zip
└── ours_wan_t2v_vipe/
    ├── 0000040.pth
    ├── 0000040.net_model/
    └── 0000040.net_model.zip
```

`pretrained_ckpt_path` / `--ckpt_path` = `checkpoints/wan_t2v_checkpoints_release/dmd2_t2v_vipe/0001000` (no
extension). The loader looks for `<prefix>.pth` plus `<prefix>.<key>_model/` directories, so the zip **must** be
unzipped and the directory **must** be named `<step>.net_model`.

## 3. What is in the checkpoints

Script: `runs/phase0/inspect_ckpt_keys.py` → `runs/phase0/inspect_ckpt_keys.log` (Slurm job 2216762);
side files: `runs/phase0/dump_pth.py` → `runs/phase0/dump_pth.log`.

| | `dmd2_t2v_vipe/0001000` | `ours_wan_t2v_vipe/0000040` |
|---|---|---|
| DCP dirs present | `net_model` only | `net_model` only |
| DCP entries / top-level prefix | 825, all `transformer.*` | 825, all `transformer.*` |
| elements / dtype | 1.4190 B, all bf16 | 1.4190 B, all bf16 |
| fake_score / discriminator / optimizer / EMA in DCP | **none** | **none** |
| vs FastGen `Wan_1_3B_Config` net (825 entries) | missing 0, unexpected 0, shape mismatch 0 | missing 0, unexpected 0, shape mismatch 0 |
| `.pth` `iteration` | 1000 | 40 |
| `.pth` LR-scheduler states | `net` (step 200), `fake_score` (step 800), `discriminator` (step 800) | `net` (step **8**), `fake_score` (step 32), **no discriminator** |
| `.pth` callbacks / grad_scaler | all empty dicts | all empty dicts |

**Answer to the Notion question: the release does NOT include critic weights.** Both checkpoints contain only the
student generator (`net`). There is no fake_score, discriminator, optimizer state or EMA.

The `.pth` scheduler states still show what the training runs looked like:
- DMD2 run: 1000 iterations = 200 generator + 800 critic updates → `student_update_freq=5`, GAN **on** (a
  discriminator scheduler exists). The critic existed but was not released.
- Official DFD run: 40 iterations = **8 generator + 32 fake_score updates**, and **no discriminator scheduler**.
  `DMD2Model.scheduler_dict` includes the discriminator iff `gan_loss_weight_gen > 0`. So the released DFD result
  was most likely trained **with the GAN term off (`gan_loss_weight_gen = 0`)**, unlike the README / config value
  0.03. (Caveat: this assumes the released code version produced that file.)
- The `.pth` files are loaded during pretrained loading, but only `iteration` is used (`model.resume_iter`, which
  only affects EMA / CT-schedule callbacks, both inactive here). Scheduler states are not restored by
  `load_pretrained_ckpt`.

Loading check (Hard rule): `net` has 0 missing / 0 unexpected keys for both checkpoints. The inference runs
loaded both checkpoints without error (`dcp.load` would have raised on missing keys).

## 4. Missing key in `pretrained_ckpt_key_map`: raise or skip?

**It raises.** For every `k_model → k_ckpt` whose model attribute exists, `FSDPCheckpointer.load` runs
`assert os.path.exists(f"{path}.{k_ckpt}_model")` (`fastgen/utils/checkpointer.py:388`). With the README map
`{net: net, fake_score: fake_score, discriminator: discriminator}` on a net-only checkpoint, `net` loads first,
then training aborts with `AssertionError: Key fake_score does not exist in FSDP model dict`.
The only silent skip is when the *model* has no such submodule. Within one key, `dcp.load` raises on keys that
the model has but the checkpoint lacks, and silently ignores extra checkpoint keys.

## 5. Consequence: the critic is missing

With the released checkpoint, the README post-train command **does not run as written**: it aborts on
`fake_score` (§4). To run it, the key map has to become `{net: net}`. Then E1 amounts to:

- `net` ← `dmd2_t2v_vipe/0001000` (the DMD2 student),
- `fake_score` ← **copy of the teacher** (Wan2.1-T2V-1.3B from HF), which never tracked the DMD2 student,
- `discriminator` ← **random init** (if `gan_loss_weight_gen > 0`),
- optimizers / LR schedulers fresh.

The authors instead continued with a fake_score that had tracked the student for 800 critic updates. Their
40-step run gave that fake_score only 32 more updates and the generator 8. A teacher-copy fake_score is therefore
a large change for such a short run (spec §5, risk 1). Open decisions: see §8 Q3 of the spec (questions sent to
the user).

## 6. Sample renders (dmd2 vs ours)

README inference command (`--do_student_sampling True`, 4 steps, 81 frames, 832×480, 16 fps), first 4 prompts of
`web_dfd_eval_prompts.txt`, `trainer.seed` ∈ {1, 2}. Both checkpoints generate without errors (~2.5 min per run
of 4 videos incl. model load on 1× A100-40GB).
Videos: `runs/phase0/samples/<ckpt>/seed<s>/prompts_first4/`. Contact sheets (rows dmd2 s1, dmd2 s2, ours s1,
ours s2; frames 0/20/40/60/80): `runs/phase0/frames/sheet_prompt{0..3}.png`.
Stats: `runs/phase0/sample_stats.py` → `sample_stats.txt` / `.jsonl` (208×120 downsampled, all 81 frames).

| metric (mean of 4 prompts × 2 seeds) | dmd2 | ours |
|---|---|---|
| HSV saturation mean | 0.527 | 0.339 |
| saturation p95 | 1.000 | 0.627 |
| fraction of pixels with saturation > 0.8 | 26.4 % | 1.1 % |
| fraction of clipped pixels (<0.02 or >0.98) | 15.3 % | 0.2 % |
| colorfulness (Hasler–Süsstrunk) | 0.244 | 0.132 |
| mean brightness | 0.277 | 0.276 |
| mean abs frame-to-frame diff | 0.023 | 0.016 |

- **Saturation / fidelity: `ours` is clearly better, consistent with the paper.** `dmd2` has crushed blacks,
  blown-out saturated colors (neon purple/pink sky, orange coffee, hyper-yellow umbrella) and very high contrast.
  `ours` has natural tones at the same average brightness.
- **Diversity: not measurable from this sample.** For a fixed seed, both models produce the same layout (same
  initial noise), and across seeds both change composition, clothing and background. The seed-1-vs-2 pixel
  difference is lower for `ours` (0.126 vs 0.239 mean), but that metric is dominated by contrast, so it is not a
  diversity measure. A diversity claim needs more seeds per prompt plus a feature-space metric / VBench (§6).
- The lower frame-to-frame diff of `ours` may partly reflect lower contrast. Motion is judged later with VBench
  Dynamic Degree / Motion Smoothness.

## 7. ViPE `long` data (downloaded 2026-09-24)

`csusupergear/vipe_fliter_complete`, `long/` only: 291 WebDataset shards + `manifest.json`, 37.2 GB, in
`data/vipe_fliter_complete/long/`. Each shard has 100 samples `<id>.mp4` + `<id>.txt` (long caption, ~700–1000
chars). Checked samples: **81 frames, 16 fps, 1280×720** (5.06 s). `VideoLoaderConfig` maps `real ← mp4`,
`condition ← txt`, `neg_condition ← preset neg_prompt_wan`. With `decoder_type="full"`, `transform_video` takes
frames `[0, sequence_length)` (**always from frame 0, no random start, no fps resampling**), resizes the short side
to 480 (→ 853×480) and center-crops to 832×480. Note for §2.5: if T < 21 is locked, the original loader
crops the *first* `sequence_length` frames. The spec's "random start for training" would need a new default-off
flag.

## 8. Environment fix found by the frame probe (2026-09-24)

`requirements.txt` leaves `wandb[media]` unpinned; pip installed wandb 0.30.0, which no longer has
`wandb.util.generate_id` (called by `fastgen/callbacks/wandb.py:107`). Training therefore crashed in
`Trainer.__init__` (first probe, job 2217295, status `error`, no memory measured). Fixed by pinning
**`wandb[media]==0.22.3`** in `.venv` (still has `generate_id`, `pip check` clean). Inference is unaffected.
