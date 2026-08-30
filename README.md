<h1 align="center">Data-Forcing Distillation (DFD)</h1>
<h3 align="center">Restoring Diversity and Fidelity in Few-Step Video Generation</h3>

<p align="center">
  Siyi Chen · Shaowei Liu · Yixuan Jia · Zian Wang · Huan Ling · Qing Qu · Jun Gao
</p>

<p align="center">
  <a href="https://csy2077.github.io/dfd-project-page/">Project Page</a> ·
  <a href="https://github.com/csy2077/DFD-self-forcing">Autoregressive (Self-Forcing) code</a>
</p>

---

This repository holds the **text-to-video (Wan2.1-1.3B)** and **image-to-video
(Cosmos-Predict2.5-2B)** experiments for **Data-Forcing Distillation (DFD)**.

DFD is a simple post-training framework that restores diversity and fidelity in
DMD-distilled few-step video generators. Standard DMD/DMD2 collapses sample
diversity and over-saturates outputs — a side effect of the mode-seeking
reverse-KL objective. DFD adds **explicit real-data supervision** to the
distribution-matching loss: it evaluates the teacher score at a *real* sample
instead of the student's own generation (the *teacher score discrepancy*). This
pulls the student toward modes it has missed and away from over-saturated modes
absent in real data — and it amounts to a **single line of code change** on top
of DMD2.

> This codebase is adapted from [NVIDIA FastGen](https://github.com/NVlabs/FastGen).
> The autoregressive (Self-Forcing) experiments live in a separate repository:
> [DFD-self-forcing](https://github.com/csy2077/DFD-self-forcing).

## The one-line change

DFD stochastically replaces the teacher-score input with a matched **real** video.
In FastGen this is exposed as a single flag, `model.gen_real_replace_prob` (the
mixing probability `p`); the rest of the DMD2 pipeline is unchanged.

```diff
- teacher_data = gen_data.detach()                                     # original DMD2
+ teacher_data = data.detach() if (rand() < p) else gen_data.detach()  # DFD
```

## Repository Structure

```
fastgen/
├── fastgen/
│   ├── callbacks/           # Training callbacks (EMA, profiling, etc.)
│   ├── configs/             # Configuration system
│   │   ├── experiments/     # Experiment configs (WanT2V, CosmosPredict2, ...)
│   │   └── methods/         # Method-specific configs
│   ├── datasets/            # Dataset loaders
│   ├── methods/             # Training methods (CM, DMD2, SFT, KD, ...)
│   ├── networks/            # Neural network architectures
│   ├── third_party/         # Third-party dependencies
│   ├── trainer.py           # Main training loop
│   └── utils/               # Utilities (distributed, checkpointing)
├── scripts/                 # Inference and evaluation scripts
├── tests/                   # Unit tests
├── Makefile                 # Development commands (lint, format, test)
└── train.py                 # Main training entry point
```

## Setup

**Recommended:** use the provided Docker container for a consistent environment
(see [CONTRIBUTING.md](CONTRIBUTING.md)). Otherwise create a conda environment:

```bash
conda create -y -n fastgen python=3.12.3 pip
conda activate fastgen
```

### Installation

```bash
git clone https://github.com/csy2077/data-forcing-distillation.git
mv data-forcing-distillation FastGen
cd FastGen
pip install -e .
```

### Credentials (optional)

For W&B logging, [get your API key](https://wandb.ai/settings) and save it to
`credentials/wandb_api.txt` or set the `WANDB_API_KEY` environment variable.

## Data

All experiments — both the Wan2.1 text-to-video and the Cosmos image-to-video runs —
use the **ViPE filtered set**:
[csusupergear/vipe_fliter_complete](https://huggingface.co/datasets/csusupergear/vipe_fliter_complete).
It builds on the existing filtering / annotation from
[nvidia/vipe-wild-sdg-1m](https://huggingface.co/datasets/nvidia/vipe-wild-sdg-1m).
The set has three folders — `long`, `medium`, `short` — by annotation (caption)
length. **All our experiments use the `long` folder.**

## Checkpoints

Released DMD2 / DFD checkpoints:

- **Wan2.1 text-to-video:**
  [csusupergear/wan_t2v_checkpoints_release](https://huggingface.co/csusupergear/wan_t2v_checkpoints_release)
- **Cosmos-Predict2.5 image-to-video:**
  [csusupergear/cosmos_i2v_checkpoints](https://huggingface.co/csusupergear/cosmos_i2v_checkpoints/tree/main/cosmos_dmd2_checkpoints)

---

## Text-to-Video (Wan2.1-1.3B)

### 1. Download the pretrained Wan2.1 DMD2 generator

Download the DMD2-pretrained Wan2.1 generator checkpoints from
[csusupergear/wan_t2v_checkpoints_release](https://huggingface.co/csusupergear/wan_t2v_checkpoints_release)
and point `pretrained_ckpt_path` at the downloaded step folder. These checkpoints
are all trained on the ViPE `long` set (see [Data](#data)).

### 2. DFD post-training

```bash
torchrun --nproc_per_node=8 train.py \
  --config fastgen/configs/experiments/WanT2V/config_dmd2.py - \
  trainer.fsdp=True \
  dataloader_train.datatags='["WDS:<your vipe long dataset folder>"]' \
  trainer.batch_size_global=16 \
  dataloader_train.shuffle_size=100 \
  dataloader_train.num_workers=2 \
  log_config.name=fastgen_postrain_reproduce \
  trainer.checkpointer.pretrained_ckpt_path="<path to checkpoints>/<checkpoints index>" \
  dataloader_train.batch_size=2 \
  trainer.logging_iter=10 \
  trainer.save_ckpt_iter=10 \
  +dataloader_train.decoder_type="full" \
  model.post_train=True \
  model.gen_real_replace_prob=0.5 \
  +'trainer.checkpointer.pretrained_ckpt_key_map={net: net, fake_score: fake_score, discriminator: discriminator}'
```

Parameters you may adjust for your setup:

- `--nproc_per_node=8` — number of GPUs (e.g. 16).
- `dataloader_train.datatags='["WDS:<your vipe long dataset folder>"]'` — point `WDS:`
  at the folder containing the ViPE `long` dataset.
- `trainer.batch_size_global=16` — keep the **global** batch size at 16.
- `dataloader_train.shuffle_size` / `num_workers` — increase if you have more CPU.
- `log_config.name` — the W&B run name.
- `trainer.checkpointer.pretrained_ckpt_path` — folder containing the checkpoint;
  **must include the step subfolder** (e.g. `0001200`).
- `dataloader_train.batch_size=2` — per-GPU batch size; raise it to use more GPU
  memory. Gradient accumulation is computed automatically from
  `batch_size_global`, GPU count, and `batch_size` — you do not set it manually.
- `trainer.logging_iter` / `trainer.save_ckpt_iter` — logging / checkpoint
  frequency (10 / 10 works well for the 128 post-training run).
- **Do not change** `model.post_train=True`, `model.gen_real_replace_prob`
  (the DFD mixing probability `p`), `decoder_type`, or the
  `pretrained_ckpt_key_map`.

Training assets (checkpoints, intermediate videos) are saved under
`FASTGEN_OUTPUT/fastgen/wan_dmd2`. Change the output root with:

```bash
export FASTGEN_OUTPUT_ROOT="<your output path>"
```

### 3. Inference

```bash
torchrun --nproc_per_node=1 --standalone \
  scripts/inference/video_model_inference.py \
  --ckpt_path <checkpoints_path>/<checkpoints_index> \
  --do_student_sampling True \
  --do_teacher_sampling False \
  --prompt_file ./web_dfd_eval_prompts.txt \
  --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
  --save_as_gif False \
  --video_save_dir <save path> - \
  trainer.seed=1 trainer.fsdp=True trainer.ddp=False \
  log_config.name=wan_posttrain_inference
```

`<checkpoints_index>` is the seven-digit step folder (e.g. `0000300`). The default
prompts are in `web_dfd_eval_prompts.txt`.

---

## Image-to-Video (Cosmos-Predict2.5-2B)

### 1. Download the Cosmos checkpoint

Set your HuggingFace token in the environment (do **not** hard-code it), then:

```bash
export HF_TOKEN=<your_hf_token>
hf download nvidia/Cosmos-Predict2.5-2B --include "base/post-trained/*" --local-dir ./
```

Point the config at the downloaded `.pt` file in
[`fastgen/configs/experiments/CosmosPredict2/config_dmd2.py`](fastgen/configs/experiments/CosmosPredict2/config_dmd2.py):

```python
config.model.pretrained_model_path = "./cosmos_predict2/Cosmos-Predict2.5-2B/base/post-trained/<id>_ema_bf16.pt"
```

### 2. (Optional) DMD2 pretraining

DFD post-training starts from a DMD2-pretrained Cosmos generator. You can skip
this step and **download the DMD2-pretrained checkpoint directly** from
[csusupergear/cosmos_i2v_checkpoints](https://huggingface.co/csusupergear/cosmos_i2v_checkpoints/tree/main/cosmos_dmd2_checkpoints).
Only run the pretraining below if you want to reproduce it from scratch.

<details>
<summary><b>Reproduce DMD2 pretraining (optional)</b></summary>

```bash
torchrun --nproc_per_node=2 --standalone train.py \
  --config fastgen/configs/experiments/CosmosPredict2/config_dmd2_v2w.py - \
  trainer.fsdp=True \
  trainer.logging_iter=10 \
  trainer.save_ckpt_iter=100 \
  model.fsdp_meta_init=True \
  dataloader_train.datatags='["WDS:<your vipe long dataset folder>"]' \
  +dataloader_train.decoder_type="full" \
  trainer.batch_size_global=<global batch size> \
  dataloader_train.shuffle_size=100 \
  dataloader_train.num_workers=2 \
  log_config.name=cosmos_pretrain_dmd2 \
  dataloader_train.batch_size=<per-gpu batch size> \
  model.gen_real_replace_prob=0 \
  trainer.fsdp_cpu_offload=False
```

Notes:

- Cosmos is sensitive to resolution and only works at **480p or 720p** (720p is
  best). The resolution is set by `config.model.input_shape` in the config, e.g.
  `config.model.input_shape = [16, 21, 88, 160]` (720p).
- You can tune **Selective Activation Checkpointing (SAC)** to trade speed for
  memory. See [`fastgen/configs/net.py`](fastgen/configs/net.py); options are
  `NONE`, `BLOCK_WISE`, `AGGRESSIVE`, `SAVE_FLASH_ATTN`, `RECOMPUTE_ALL`. Defaults
  are `BLOCK_WISE` for the generator and `AGGRESSIVE` for the teacher:
  ```python
  config.model.net     = CosmosPredict2_2B_Aggressive_Config
  config.model.teacher = CosmosPredict2_2B_Recompute_all_Config
  ```

</details>

### 3. DFD post-training (on top of the DMD2 checkpoint)

```bash
torchrun --nproc_per_node=8 train.py \
  --config fastgen/configs/experiments/CosmosPredict2/config_dmd2_v2w.py - \
  trainer.fsdp=True \
  dataloader_train.datatags='["WDS:<your vipe long dataset folder>"]' \
  trainer.batch_size_global=128 \
  dataloader_train.shuffle_size=100 \
  dataloader_train.num_workers=2 \
  log_config.name=fastgen_postrain_cosmos_i2v \
  trainer.checkpointer.pretrained_ckpt_path="<your saving folder>/checkpoints/0003600" \
  dataloader_train.batch_size=2 \
  trainer.logging_iter=10 \
  trainer.save_ckpt_iter=10 \
  +dataloader_train.decoder_type="full" \
  model.post_train=True \
  model.gen_real_replace_prob=0.5 \
  +'trainer.checkpointer.pretrained_ckpt_key_map={net: net, fake_score: fake_score, discriminator: discriminator}'
```

### 4. Inference (ViPE test set)

The ViPE test set is included in this repo under
[`vipe_testset_data/`](vipe_testset_data) (conditioning images plus
`source_image_paths.txt` and `caption.txt`), so you can run inference directly:

```bash
torchrun --nproc_per_node=1 --standalone \
  scripts/inference/video_model_inference_vbench.py \
  --config fastgen/configs/experiments/CosmosPredict2/config_dmd2_v2w.py \
  --do_student_sampling True \
  --do_teacher_sampling False \
  --num_steps 35 --fps 24 \
  --ckpt_path <checkpoints_path>/<checkpoints_index> \
  --neg_prompt_file scripts/inference/prompts/negative_prompt_cosmos.txt \
  --input_image_file vipe_testset_data/source_image_paths.txt \
  --num_conditioning_frames 1 \
  --save_as_gif False \
  --video_save_dir <save path> \
  --prompt_file vipe_testset_data/caption.txt - \
  model.net.is_video2world=True \
  trainer.seed=1 trainer.fsdp=True trainer.ddp=False
```

---

## Citation

```bibtex
@misc{chen2026dataforcingdistillationrestoringdiversity,
      title={Data-Forcing Distillation: Restoring Diversity and Fidelity in Few-Step Video Generation}, 
      author={Siyi Chen and Shaowei Liu and Yixuan Jia and Zian Wang and Huan Ling and Qing Qu and Jun Gao},
      year={2026},
      eprint={2606.18478},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.18478}, 
}
```

## Acknowledgements

This codebase is adapted from [NVIDIA FastGen](https://github.com/NVlabs/FastGen).
The Cosmos and Wan2.1 base models are from NVIDIA and the Wan-Video team,
respectively.
