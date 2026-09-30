# Workspace migration guide (DFD real-data experiments)

Written 2026-09-30 on Sulis (`/gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation`) to move the project to
another server. It covers: what the workspace contains, how to rebuild the environments, what to copy vs.
re-download vs. regenerate, the experiment plan, what has been done, and what is still open.

Read order on the new server: this file → `Claude.md` → `Dfd experiments.md` (full spec, called `DFD_EXPERIMENTS.md`
in `Claude.md`) → `RESULTS.md` → `runs/PHASE0.md`, `runs/NOTES_code_reading.md`, `runs/FRAMES_LOCK.md`,
`runs/NOTES_paper.md`, `runs/data_prep/NOTES.md`, `runs/<EXP_ID>/NOTES.md`.

---

## 1. The project in one paragraph

Start from DFD's released Wan2.1-T2V-1.3B 4-step DMD2 student (`csusupergear/wan_t2v_checkpoints_release/dmd2_t2v_vipe/0001000`),
reproduce DFD post-training (40 steps on synthetic ViPE `long` videos, which are Wan2.1-14B generations), then study
why swapping in real-world video (Mixkit) removes DFD's diversity gain, and test fixes: E3 (CFG variant), E4
(teacher-ODE inversion noise), E5 (new loss, formula pending from the user), with F2 diagnostics of the theory's
constants. Base code: FastGen fork `csy2077/data-forcing-distillation`, commit `de6416c`. No FastWan.

---

## 2. Workspace layout

Sizes as of 2026-09-30.

```
data-forcing-distillation/            repo root (git, base commit de6416c; ALL our changes are uncommitted, see §5)
├── Claude.md                         working rules + current status (Claude updates it after each step)
├── Dfd experiments.md                full experiment spec; [VERIFIED]/[CONFIRMED]/[ASK ME] tags
├── RESULTS.md                        results table (spec §7) + dated notes
├── MIGRATION.md                      this file
├── env/                              frozen package lists of the 3 venvs (uv pip freeze, 2026-09-30)
│   ├── venv.freeze.txt               training / inference / inversion / extra metrics
│   ├── venv-vlm.freeze.txt           Mixkit prep + Qwen2.5-VL captioning + WDS building
│   └── venv-vbench.freeze.txt        VBench scoring
├── fastgen/                          FastGen library (4 files modified, §5)
├── scripts/
│   ├── train_dfd.py                  = train.py + per-submodule load report (load_report.json)
│   ├── launch_run.sh                 writes runs/<EXP_ID>/{cmd.sh,git_diff.patch,train.log,config_dump.yaml,load_report.json}
│   ├── probe_frames.py               memory probe (spec §2.5)
│   ├── invert_real.py                E4 teacher-ODE inversion
│   ├── data_prep/                    prep_mixkit.py, caption_qwen.py, build_wds.py, build_inv_wds.py, caption checks
│   ├── eval/                         VBench subset generation/scoring, extra metrics, bootstrap, trajectory, diag summary
│   └── inference/                    original FastGen inference (unchanged)
├── tests/test_dfd_flags.py           flags-off == original code bit-exactly; flag behavior
├── tests/test_full_save_iter.py      full_save_iter checkpoint flag
├── runs/                             9.5 GB — ALL experiment state, logs, notes, eval videos and metrics (§3: copy it)
│   ├── PHASE0.md, NOTES_code_reading.md, FRAMES_LOCK.md, NOTES_paper.md
│   ├── sbatch/                       Slurm scripts per run; common.sh (E1 overrides), e2_common.sh (E2–E4 overrides)
│   ├── <EXP_ID>/                     E1a_seed1, E1d_seed1, E2_seed1, E2s_seed1, E3_seed1, E3s_seed1, E4_seed1, SMOKE_*
│   ├── eval/                         8.8 GB: per-model 800 subset videos + VBench/extra metrics; vbench_subset/ (fixed prompts)
│   ├── data_prep/                    Mixkit pipeline logs + caption validation notes
│   ├── probe_frames*/                memory probes
│   └── phase0/, compare/, smoke/     Phase 0 samples, side-by-side comparison videos
├── data/                             (§3 says which ones to move)
│   ├── vipe_fliter_complete/long/    35 GB  ViPE long WDS (HF csusupergear/vipe_fliter_complete, long/ only)
│   ├── mixkit_src/                   24 GB  HF FastVideo/Mixkit-Src @ b5fff04254af8bbf716b1c166e85b8fe4000bb0e
│   ├── mixkit_prep/                  11 GB  clips/ (5,227 × 81 f, 1280×720, 16 fps) + meta.jsonl + captions_v3.jsonl (+ rejected v2)
│   ├── mixkit_wds_v3/                11 GB  E2 / E3 data (53 shards: .mp4 + .txt)
│   ├── mixkit_vipe_pair_wds_v3/      17 GB  E2s / E3s data (+ .student.mp4 ViPE partner + .pair.json)
│   ├── mixkit_inv/                   11 GB  E4 inversion output: z/<id>.npy, meta_s*.jsonl, subset1280.txt
│   ├── mixkit_inv_wds_v3/            13 GB  E4 data (13 shards: .mp4 + .txt + .z_inv.npy + .json), built 2026-09-30
│   ├── mixkit_wds/, mixkit_vipe_pair_wds/  11+17 GB  OBSOLETE (v2 captions, rejected) — do not move
│   ├── mixkit_*_test/, caption_val/  small test / caption-validation sets
│   └── vbench_cache/                 200 MB VBench model weights (aesthetic, amt, pyiqa, raft)
├── checkpoints/wan_t2v_checkpoints_release/   11 GB  dmd2_t2v_vipe/0001000 + ours_wan_t2v_vipe/0000040 (zips + unzipped DCP dirs)
├── FASTGEN_OUTPUT/                   384 GB
│   ├── .cache/ (= HF_HOME)           50 GB  Wan2.1-T2V-1.3B-Diffusers, Qwen2.5-VL-7B-Instruct, all-mpnet-base-v2, DINOv2 ViT-B/14, (Mixkit/ckpt HF caches)
│   └── fastgen/wan_dmd2/<run>/       trained checkpoints: E1a/E2/E2s/E3/E3s_seed1 64 GB each, E1d 16 GB; SMOKE_* logs only
├── .venv/ (7.6 GB), .venv-vlm/, .venv-vbench/ (6.8 GB), .uv-python/   — rebuild, don't copy (§4)
└── vipe_testset_data/, web_dfd_eval_prompts.txt   from the original repo
```

Trained-checkpoint layout (`FASTGEN_OUTPUT/fastgen/wan_dmd2/E2_seed1/checkpoints/`): `00000{05..35}.net_model` +
`.pth` are net-only (2.7 GB each, enough for inference); `0000020`, `0000040`, `0000041` are full (net + fake_score
+ both optimizers, ≈ 16 GB each). `trainer.max_iter=41` gives 40 training iterations (8 generator updates,
`student_update_freq=5`).

---

## 3. What to move

### 3.1 Copy (expensive or impossible to reproduce exactly)

| What | Size | Why |
|---|---|---|
| Repo code incl. uncommitted changes (`fastgen/`, `scripts/`, `tests/`, top-level `.md`, `env/`) | < 50 MB | our implementation; commit it on a branch first (§5) or rsync it |
| `runs/` (whole) | 9.5 GB | all notes, logs, F2 diagnostics, **eval videos of E0/REF/E1a/E2/E2s/E3/E3s** (needed as paired baselines for E4 and later; regenerating costs ~2 h × 2 GPUs per model) |
| `data/mixkit_prep/captions_v3.jsonl`, `meta.jsonl`, `caption_list.txt` | 7 MB | Qwen captions (sampled, T=0.7; per-clip seeds stored, but re-running is ~1 h × 2 GPUs and not guaranteed bit-identical across hardware) |
| `data/mixkit_inv/` (or at least `z/`, `meta_s*.jsonl`, `subset1280.txt`) | 11 GB | E4 inversion = **2 × 13 h on 2 A100 each** |
| `data/mixkit_wds_v3/`, `data/mixkit_vipe_pair_wds_v3/`, `data/mixkit_inv_wds_v3/` | 11 + 17 + 13 GB | ready-to-train data; rebuilding needs mixkit_prep clips + the steps in §6.4 |
| `runs/eval/vbench_subset/` | small (inside runs/) | fixed 160-prompt subset; every model must use the same one |

Optional copies: `data/mixkit_prep/clips/` (11 GB, only needed to rebuild the WDS or re-caption), the trained
checkpoints (only needed to re-evaluate or to start something from them; if wanted, copy just
`<run>/checkpoints/0000020.net_model`, `0000040.net_model` and their `.pth`, ≈ 5.4 GB per run),
`data/vbench_cache/` (200 MB, avoids re-downloading VBench weights).

### 3.2 Re-download on the new server

| What | Source | Where it must end up |
|---|---|---|
| Teacher Wan2.1-T2V-1.3B | HF `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (28.9 GB) | `$HF_HOME` (= `FASTGEN_OUTPUT/.cache`) |
| DMD2 start + official DFD result | HF `csusupergear/wan_t2v_checkpoints_release`, folders `dmd2_t2v_vipe/`, `ours_wan_t2v_vipe/` | `checkpoints/wan_t2v_checkpoints_release/…`; **unzip** `0001000.net_model.zip` → dir `0001000.net_model/` next to `0001000.pth` (loader needs `<prefix>.pth` + `<prefix>.net_model/`; `runs/PHASE0.md` §2) |
| ViPE `long` | HF dataset `csusupergear/vipe_fliter_complete`, `long/` only (291 shards, 37 GB) | `data/vipe_fliter_complete/long/` |
| Mixkit source (only if clips are not copied) | HF dataset `FastVideo/Mixkit-Src`, **revision `b5fff04254af8bbf716b1c166e85b8fe4000bb0e`** | `data/mixkit_src/` |
| Qwen2.5-VL-7B-Instruct, `sentence-transformers/all-mpnet-base-v2` (only to re-caption / re-pair) | HF | `$HF_HOME` |
| DINOv2 ViT-B/14 (`timm/vit_base_patch14_dinov2.lvd142m`) | HF via timm, fetched on first use | `$HF_HOME` |
| VBench weights (if `data/vbench_cache` not copied) + official `VBench_full_info.json` | vbench 0.1.5 downloads on first run | `data/vbench_cache/`; subset sha256 is in `runs/eval/vbench_subset/subset_meta.json` |

Hint: use an HF token for bulk downloads (anonymous downloads hit HTTP 429).

### 3.3 Don't move

`.venv*`, `.uv-python` (rebuild), `data/mixkit_wds/`, `data/mixkit_vipe_pair_wds/` (v2 captions, rejected),
`FASTGEN_OUTPUT/fastgen/wan_dmd2/{vbench_*,phase0_*,invert_*}` (only logs/configs of inference runs), `__pycache__`.

### 3.4 Example rsync (run from the new server)

```bash
SRC=sulis:/gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
DST=/path/to/data-forcing-distillation
mkdir -p $DST/data
rsync -avP --exclude '.venv*' --exclude '.uv-python' --exclude '/FASTGEN_OUTPUT' --exclude '/checkpoints' \
      --exclude '/data' --exclude '__pycache__' $SRC/ $DST/          # code (incl. .git, base de6416c) + runs/ + notes
for d in mixkit_wds_v3 mixkit_vipe_pair_wds_v3 mixkit_inv_wds_v3 mixkit_inv vbench_cache; do
  rsync -avP $SRC/data/$d $DST/data/; done
rsync -avP --exclude clips $SRC/data/mixkit_prep $DST/data/          # captions + meta (add clips/ if you want to rebuild)
```

---

## 4. Environment setup

Hardware used: 1 node, **2× A100-PCIE-40GB**, driver 560 (CUDA 12.6). Training peak 30.2 GB/GPU at per-GPU batch 1
(76 % of 40 GB), host RAM ≈ 32 GB per rank. A 40 GB GPU is the minimum at T=21; with batch 2 it OOMs.

### 4.1 Training env `.venv` (Python 3.12.7)

```bash
cd $DST
python3.12 -m venv .venv && source .venv/bin/activate
pip install uv
# torch must match the driver: cu126 wheels for a CUDA 12.6 driver; pick the matching index for a newer driver
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install -e .
uv pip install pandas "wandb[media]==0.22.3" pytest "setuptools<81"
# exact versions of everything else: env/venv.freeze.txt (drop the "-e file://..." line and the +cu126 torch lines)
```

Gotchas (from `runs/PHASE0.md` §1/§8):
- `pandas` is imported unconditionally but missing from `requirements.txt`.
- `wandb` must be **0.22.3**: 0.30 removed `wandb.util.generate_id`, and training crashes in `Trainer.__init__`.
- `transformers==4.49.0`, `diffusers==0.35.1`, `numpy==1.26.4`.
- diffusers 0.35.1 calls the Hub API when loading the sharded Wan transformer even with `HF_HUB_OFFLINE=1`, so leave
  offline mode off (compute nodes need internet), or patch around it if the new cluster is offline.
- Append to `.venv/bin/activate` (the scripts assume it):
  ```bash
  _DFD_ROOT="$DST"
  export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-$_DFD_ROOT/FASTGEN_OUTPUT}"
  export HF_HOME="$_DFD_ROOT/FASTGEN_OUTPUT/.cache"
  ```

### 4.2 Data/caption env `.venv-vlm` (Python 3.12)

Only needed to re-run Mixkit prep, captioning or WDS building. torch 2.8.0+cu126, `transformers==4.57.1`,
`qwen-vl-utils==0.0.14`, `scenedetect==0.7.1`, `av==18.1.0` (PyAV wheels bundle FFmpeg with libx264, so no system
ffmpeg is needed), `sentence-transformers==5.7.0`, `webdataset==1.0.2`; full list `env/venv-vlm.freeze.txt`.

### 4.3 VBench env `.venv-vbench` (Python 3.10)

```bash
uv python install 3.10 && uv venv --python 3.10 .venv-vbench
uv pip install --python .venv-vbench/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-vbench/bin/python vbench==0.1.5 transformers==4.33.2   # rest: env/venv-vbench.freeze.txt
```
`scripts/eval/run_vbench.py` sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (torch ≥ 2.6 breaks VBench's AMT load) and uses
`VBENCH_CACHE_DIR=data/vbench_cache`.

### 4.4 Paths and scheduler to adapt

- The absolute path `/gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation` is hard-coded in ~45 files
  (`scripts/launch_run.sh`, `scripts/eval/*.sh|*.sbatch`, `runs/sbatch/*`, `runs/*/*.sbatch`, and the recorded `cmd.sh`
  files). Replace it in the *scripts* you will run; leave historic `runs/<EXP_ID>/cmd.sh` as a record:
  ```bash
  grep -rl /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation scripts runs/sbatch runs/eval/*.sbatch runs/E4_seed1/*.sbatch \
    | xargs sed -i "s#/gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation#$DST#g"
  ```
- Slurm headers are Sulis-specific: `-A su007-hw-gpu`, `-p gpu`, `--gres=gpu:ampere_a100:2`, `--mem-per-cpu=3850`.
  Change them for the new cluster (or run the bodies directly without Slurm). Sulis also forbids CPU-only jobs and caps
  login-node memory at ~540 MB; those rules can be dropped if the new server has no such limits (update `Claude.md`).
- W&B runs offline (`WANDB_MODE=offline`, `log_config.wandb_mode=offline`).

### 4.5 Keep the locked settings

Keep **T=21 (81 frames), per-GPU batch 1, global batch 16** (`runs/FRAMES_LOCK.md`, [CONFIRMED]) even if the new GPUs
have more memory: per-GPU batch sets the granularity of DFD's all-or-nothing replacement (`runs/NOTES_code_reading.md` §2),
so changing it breaks comparability with every existing row. More GPUs are fine: FastGen sets grad accumulation =
16 / (NGPU × 1). Changing the GPU count (2 → N) changes data order and reduction order, so new runs are comparable
in distribution, not bit-identical; say so in the notes.

---

## 5. Code changes vs. base commit `de6416c`

**Not committed.** `git status` shows 4 modified library files plus untracked scripts/tests/notes. Before moving,
commit them on a branch (e.g. `dfd-real-data`) or rsync the tree. `runs/<EXP_ID>/git_diff.patch` holds the exact
diff each run used.

All changes are behind default-off flags; flags off = original DFD bit-exactly (`tests/test_dfd_flags.py`).

| Flag (config) | Default | Used by | What it does | File |
|---|---|---|---|---|
| `trainer.checkpointer.full_save_iter` | 0 | all post-train runs (`=20`) | saves not divisible by N (and not the final one) are net-only | `fastgen/configs/config.py`, `fastgen/trainer.py` |
| `model.dfd_student_input_key` | `""` | E2s, E3s (`student_real`) | multi-step student input from a 2nd video stream (paired ViPE clip) instead of `data["real"]` | `config_dmd2.py`, `dmd2.py` |
| `model.dfd_diagnostics_every` / `_dir` / `_sketch_dim` | 0 / "" / 65536 | E1d, E2–E4 (`=1`) | F2 no-grad diagnostics (reg_mean_norm/reg_var via count sketches, delta_sq, lipschitz_hat) → `runs/<EXP_ID>/diag/` | same |
| `model.dfd_cfg_uncond_on_gen` | False | E3, E3s | on replaced batches, CFG uncond branch at the noised generated point | same |
| `model.dfd_inverted_noise_key` | `""` | E4 (`z_inv`) | `eps_student ← data["z_inv"]` (teacher-ODE inversion) | same |

Run overrides live in `runs/sbatch/common.sh` (E1: README command + decisions: `gan_loss_weight_gen=0`, key map
`{net: net}` by default, batch 1, ViPE) and `runs/sbatch/e2_common.sh` (`DATA_E2`, `DATA_E2S`, `E3FLAG`, `DATA_E4`,
`diag()`). The E2s stream needs `++dataloader_train.key_map.student_real=student.mp4 '++dataloader_train.video_keys=[real,student_real]'`;
E4 needs `++dataloader_train.key_map.z_inv=z_inv.npy`.

Not implemented yet: E5 loss hook (`model.dfd_loss_variant`), E4b OT (`model.dfd_minibatch_ot`), F2 `B_hat` (JVP) and
`cos(g_DFD, g_DMD)`.

---

## 6. How to run things

### 6.1 Training run
```bash
# inside an allocation with 2 GPUs, from the repo root
source runs/sbatch/e2_common.sh
scripts/launch_run.sh <EXP_ID> 2 "${COMMON_NODATA[@]}" "${DATA_E4[@]}" $(diag <EXP_ID>) trainer.seed=1 \
  trainer.max_iter=41 trainer.save_ckpt_iter=5 trainer.checkpointer.full_save_iter=20 trainer.logging_iter=5 \
  trainer.checkpointer.pretrained_ckpt_path=$CK log_config.name=<EXP_ID>
```
See `runs/sbatch/<EXP_ID>.sbatch` for each run. ≈ 2 min/iteration → ≈ 1.5 h per 40-step run on 2× A100-40GB.
Smoke test first (rule): same command with `trainer.max_iter=20 trainer.save_ckpt_iter=10`, EXP_ID `SMOKE_<X>`;
check finite losses, `load_report.json` (no missing keys), sample renders; then delete `SMOKE_*/checkpoints/`
(standing permission; keep logs/config/load_report/wandb).

### 6.2 Memory probe
`runs/probe_frames_E4/probe.sbatch` → `scripts/probe_frames.py --ngpu 2 --ladder 21 --batch_sizes 1 …`. Pass criterion:
peak `max_memory_allocated` ≤ 90 % of GPU memory. Re-probe on the new GPUs before the first run there.

### 6.3 Evaluation (light eval = checkpoints it20 and it40)
```bash
sbatch --export=ALL,CKPT=FASTGEN_OUTPUT/fastgen/wan_dmd2/E4_seed1/checkpoints/0000040,TAG=E4_seed1_it0040,T=21 scripts/eval/eval_ckpt.sbatch
python3 scripts/eval/compare_to_bases.py --tags E4_seed1_it0020 E4_seed1_it0040 --bases E0_dmd2_T21 REF_ours_T21 E1a_seed1_it{step} E2_seed1_it{step}
.venv/bin/python scripts/eval/diag_summary.py runs/E2_seed1/diag runs/E4_seed1/diag --out runs/eval/diag_E2_E4.md   # needs torch
```
Pipeline: 160 prompts × seeds 0–4 = 800 videos (≈ 1 h 50 min on 2 GPUs) → VBench 7 quality dims → Quality Score
(official weights) → saturation + DINOv2 seed diversity → paired bootstrap over prompts vs. the bases.
**Migration check:** regenerate a few E0 videos (or the whole E0 subset) on the new server and compare with
`runs/eval/E0_dmd2_T21` (same seeds → should be near-identical; a different GPU type/driver can cause small
numerical differences). If the difference is noticeable, regenerate E0/REF/E1a/E2 videos on the new server before
comparing new runs with them.

### 6.4 Mixkit data (only if rebuilding)
`runs/data_prep/mixkit_pipeline.sbatch` with `PV=v3 SUFFIX=_v3` (the script's default `PV=v2` is the rejected version):
prep (`prep_mixkit.py`: drop portrait, 1920×1080 → 1280×720 AREA, 30 → 16 fps by nearest-frame selection, middle 81
frames, PySceneDetect ContentDetector thr 27 → drop cuts, libx264 crf 18 preset medium yuv420p) → captions
(`caption_qwen.py --prompt_version v3`, Qwen2.5-VL-7B, T=0.7, top-p 0.8) → `build_wds.py` (both E2 and E2s WDS;
E2s pairing = all-mpnet-base-v2 caption nearest neighbor in ViPE `long`). E4: `runs/E4_seed1/invert.sbatch`
(`LIST=data/mixkit_inv/subset1280.txt OUT=data/mixkit_inv`, 50 Euler steps, w=1, resumable) → `build_inv_wds.py`
(no torch). Details: `runs/data_prep/NOTES.md`, `runs/E4_seed1/NOTES.md`.

---

## 7. Experiment plan and status

Shared settings (spec §2.4): start `dmd2_t2v_vipe/0001000`; `post_train=True`, `gen_real_replace_prob=0.5`, lr 1e-5,
`gan_loss_weight_gen=0`, 4-step student `t_list=[0.999, 0.937, 0.833, 0.624, 0]`, 40 steps, seed 1, T=21,
global batch 16 (1 × accum 8 × 2 GPUs); fake_score = teacher copy (released ckpt is net-only). Heavy eval (all 8
checkpoints, multiple seeds) only for E0/E1; everything later = single seed, light eval (it20/it40).

| ID | What | Status | Where |
|---|---|---|---|
| Phase 0 | env, ckpt inspection (net-only, no critic), README key map raises → `{net: net}`, sample renders | **done** | `runs/PHASE0.md` |
| Frame probe | T=21 / batch 1 fits (30 GB, 76 %) | **done**, locked 2026-09-25 | `runs/FRAMES_LOCK.md` |
| E0 / REF | eval of `dmd2_t2v_vipe/0001000` and `ours_wan_t2v_vipe/0000040` | **done** | `runs/eval/E0_dmd2_T21`, `REF_ours_T21` |
| E1a_seed1 | DFD reproduction on ViPE `long`, full 8-checkpoint eval | **done**; accepted as baseline 2026-09-27 (sat gap to REF attributed to data) | `runs/E1a_seed1`, `runs/eval/trajectory_E1a_seed1.md` |
| E1b | fake_score warm-up variant (200 critic-only iters) | **deferred** (scripts kept: `runs/sbatch/E1b_*.sbatch`) | |
| E1 seeds 2–3 | reproduction noise estimate | **not done** (user: not for now) | |
| E1d_seed1 | E1a + F2 diagnostics (synthetic F2 reference, no eval) | **done** | `runs/E1d_seed1` |
| Mixkit data | 5,227 clips, Qwen v3 captions, E2/E2s WDS | **done** | `runs/data_prep/NOTES.md` |
| E2_seed1 | real data (Mixkit) in both places | **done** (it20/it40 eval + F2) | RESULTS.md |
| E2s_seed1 | ViPE student input + Mixkit replacement | **done** | RESULTS.md |
| E3_seed1 / E3s_seed1 | CFG uncond at gen point (both arms) | **done** | RESULTS.md |
| E4 inversion | 1,280-clip subset, 50 steps, w=1; recon 32.2 dB (VAE 36.5), z_std 0.85 | **done** 2026-09-29 | `runs/E4_seed1/NOTES.md` |
| E4 WDS | `data/mixkit_inv_wds_v3` | **done** 2026-09-30 | |
| E4 probe → SMOKE_E4 → E4_seed1 → eval it20/it40 + F2 vs E2 | | **not done** (jobs 2225080/2225081 were queued on Sulis, not started) | `runs/probe_frames_E4/probe.sbatch`, `runs/sbatch/{SMOKE_E4,E4_seed1}.sbatch` |
| E5 | new loss | **not done**: formula pending from the user; only prepare a hook `model.dfd_loss_variant` (not yet written) | spec §3 E5 |
| Main real dataset (Vchitect / OpenVidHD / Tiger200K) | scale E2 beyond Mixkit | **deferred** [ASK ME] | spec §2.3 |
| E2b (data cleaning / cluster centers) | optional | **not done** [ASK ME] | spec §3 |
| E4b (minibatch OT, exact vs approx) | optional | **not done** [ASK ME] | spec §3 |
| F1 (other settings: 2-step, multi-step post-train, I2V/AR) | | **not done** [ASK ME] | spec §4 |
| F2 `B_hat`, `cos(g_DFD, g_DMD)` | | **not implemented** (scope decision 2026-09-28) | spec §4 |
| Main-table motion metric (Motion Smoothness vs Dynamic Degree) | | open [ASK ME]; both reported | spec §8 Q10 |

### 7.1 Results so far (single seed after E1; `*` = prompt-level 95 % CI excludes 0)

| model | Quality | Dyn. Deg. | sat>0.8 % | div DINOv2 |
|---|---|---|---|---|
| E0 | 0.8479 | 0.820 | 32.8 | 0.2951 |
| REF (official DFD) | 0.8499 | 0.860 | 3.0 | 0.3020 |
| E1a it40 | 0.8500 | 0.805 | 11.6 | **0.3093** (+0.014* vs E0) |
| E2 it40 | 0.8534 | 0.920 | 9.1 | 0.2908 (−0.019* vs E1a) |
| E2s it40 | 0.8469 | 0.905 | 11.0 | 0.2962 (≈ E0) |
| E3 it40 | 0.8452 | 0.825 | 9.3 | 0.2945 |
| E3s it40 | 0.8427 | 0.725 | 19.5 | 0.2714 (below E0*) |

Take-aways: VBench Quality does not separate models (hence the saturation and diversity metrics). Real data
(E2) keeps DFD's de-saturation and adds motion but loses the diversity gain; putting real data only on the teacher
side (E2s) doesn't restore it; the E3 CFG variant doesn't help and is harmful on the E2s arm. F2: E2 ≈ E1d; E2s/E3s
have ≈ 4× delta_sq. Full tables and discussion: `RESULTS.md`.

---

## 8. First steps on the new server

1. Rebuild envs (§4), fix paths/Slurm headers (§4.4), download teacher/checkpoints/ViPE (§3.2), copy §3.1.
2. `pytest tests/test_full_save_iter.py tests/test_dfd_flags.py -v` on a GPU node.
3. Migration check: regenerate E0 subset videos (or a subset of seeds) and compare with `runs/eval/E0_dmd2_T21` (§6.3).
4. E4: memory probe → SMOKE_E4 (then delete its checkpoints) → E4_seed1 (40 steps, F2 on) → eval it20/it40 →
   compare with E0/REF/E1a/E2, delta_sq vs E2 → write RESULTS.md + `runs/E4_seed1/NOTES.md` conclusion.
5. E5: wait for the user's formula; add the `model.dfd_loss_variant` hook at the `variational_score_distillation_loss` call site.
6. Update the "Current status" block in `Claude.md` and replace the Sulis-specific hard rules if they don't apply.
