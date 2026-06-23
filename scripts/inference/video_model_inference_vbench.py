# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VBench-I2V inference: identical to video_model_inference.py except for the
output filenames.

VBench-I2V's `--mode=custom_input` requires:
    <video_save_dir>/<image_stem>-<idx>.mp4

where <image_stem> is the input image's basename without extension and <idx>
is the sample index (0..4 per VBench convention; pass --video_index N when
running with N different seeds). The prompt-name subdirectory used by the
default inference script is also dropped so the output dir stays flat.

Run identically to scripts/inference/video_model_inference.py — see that file
for examples. Add `--video_index <int>` (default 0) and (optionally) bump
trainer.seed across runs to fill out 0..4.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import imageio.v3 as iio
import numpy as np
import torch
from tqdm.auto import tqdm

from fastgen.configs.config import BaseConfig
from fastgen.networks.WanI2V import WanI2V
from fastgen.networks.cosmos_predict2 import CosmosPredict2
import fastgen.utils.logging_utils as logger
from fastgen.utils.distributed import clean_up, is_rank0, world_size
from fastgen.utils import basic_utils
from fastgen.utils.scripts import parse_args, setup
from fastgen.third_party.wan_prompt_expand.prompt_expand import QwenPromptExpander
from fastgen.datasets.wds_dataloaders import transform_video
from scripts.inference.inference_utils import (
    expand_path,
    load_prompts,
    init_model,
    init_checkpointer,
    load_checkpoint,
    cleanup_unused_modules,
    setup_inference_modules,
    add_common_args,
)
from scripts.inference.video_model_inference import (
    load_video_frames,
    load_conditioning_image,
    prepare_wani2v_condition,
    prepare_cosmos_v2w_condition,
    prepare_vacewan_condition,
    prepare_i2v_condition,
    expand_prompts_with_qwen,
)

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


def main(args, config: BaseConfig):
    # Load prompts
    pos_prompt_set = load_prompts(args.prompt_file, relative_to="cwd")

    # Prompt expansion if specified
    if args.prompt_expand_model:
        pos_prompt_set = expand_prompts_with_qwen(
            pos_prompt_set,
            args.prompt_expand_model,
            torch.device(config.model.device),
            args.prompt_expand_model_seed,
        )

    # Load depth latent paths
    depth_latent_paths = None
    if args.depth_latent_file is not None:
        depth_latent_path = expand_path(args.depth_latent_file, relative_to="cwd")
        if depth_latent_path.is_file():
            with depth_latent_path.open("r") as f:
                depth_latent_paths = [line.strip() for line in f.readlines()]
        else:
            raise FileNotFoundError(f"depth_latent_file: {depth_latent_path} not found!")

    # Load source video paths
    source_video_paths = None
    if args.source_video_file is not None:
        source_video_path = expand_path(args.source_video_file, relative_to="cwd")
        if source_video_path.is_file():
            with source_video_path.open("r") as f:
                source_video_paths = [line.strip() for line in f.readlines()]
        else:
            raise FileNotFoundError(f"source_video_path: {source_video_path} not found!")

    # Load input images for I2V mode (or video2world mode in cosmos)
    input_image_paths = None
    if args.input_image_file is not None:
        input_image_file_path = expand_path(args.input_image_file, relative_to="cwd")
        if input_image_file_path.is_file():
            with input_image_file_path.open("r") as f:
                input_image_paths = [line.strip() for line in f.readlines() if line.strip()]

            num_prompts = len(pos_prompt_set)
            num_images = len(input_image_paths)
            if num_images < num_prompts:
                last_image = input_image_paths[-1] if input_image_paths else ""
                input_image_paths.extend([last_image] * (num_prompts - num_images))
                logger.info(f"I2V: extended {num_images} images to {num_prompts} by repeating last image")
            elif num_images > num_prompts:
                input_image_paths = input_image_paths[:num_prompts]
                logger.info(f"I2V: truncated {num_images} images to {num_prompts} prompts")

            logger.info(f"I2V mode: {len(input_image_paths)} input images for {num_prompts} prompts")
        else:
            raise FileNotFoundError(f"input_image_file_path: {input_image_file_path} not found!")

    if input_image_paths is None:
        raise ValueError(
            "video_model_inference_vbench.py requires --input_image_file (VBench-I2V "
            "names outputs by image stem)."
        )

    # Fix sampling seeds
    seed = basic_utils.set_random_seed(config.trainer.seed, by_rank=True)

    # Initialize model and checkpointer
    model = init_model(config)
    checkpointer = init_checkpointer(config)

    # Load checkpoint
    ckpt_iter, save_dir = load_checkpoint(checkpointer, model, args.ckpt_path, config)

    if ckpt_iter is None and args.do_student_sampling:
        logger.warning(f"Performing {model.config.student_sample_steps}-step generation on the non-distilled model")

    if args.video_save_dir:  # overwrite the save_dir
        save_dir = args.video_save_dir
        logger.info(f"video_save_dir: {save_dir}")

    save_dir = Path(save_dir)
    # NOTE: VBench expects a flat folder of <image_stem>-<idx>.mp4 files, so
    # we deliberately skip the prompt_name / _hq subdirectory nesting that the
    # default inference script applies.
    save_dir.mkdir(parents=True, exist_ok=True)

    save_video_kwargs = {"precision_amp": model.precision_amp_infer, "save_as_gif": args.save_as_gif, "fps": args.fps}
    if args.save_high_quality:
        save_video_kwargs = {
            "quality": 18,
            "preset": "medium",
            "fps": args.fps,
        }

    # Remove unused modules
    cleanup_unused_modules(model, args.do_teacher_sampling)

    # Get precision and set up inference modules
    teacher, student, vae = setup_inference_modules(
        model, config, args.do_teacher_sampling, args.do_student_sampling, model.precision
    )
    ctx = {"dtype": model.precision, "device": model.device}

    has_teacher_sampling = teacher is not None and hasattr(teacher, "sample")
    has_student_sampling = student is not None and hasattr(model, "generator_fn")
    assert (
        has_teacher_sampling or has_student_sampling
    ), "At least one of teacher or student (with generator_fn) must be provided for sampling"

    if has_teacher_sampling and has_student_sampling:
        logger.warning(
            "Both student and teacher sampling enabled — VBench naming would clash; "
            "teacher outputs will be written under <video_save_dir>/teacher/ to avoid overwrite."
        )

    # Load negative condition
    neg_condition = None
    if args.neg_prompt_file is not None:
        neg_condition = load_prompts(args.neg_prompt_file, relative_to="cwd")
        if len(neg_condition) > 0:
            neg_condition = neg_condition[:1]
            logger.warning(f"Found {len(neg_condition)} negative prompts, only using the first one.")
        logger.debug(f"Loaded negative prompt: {neg_condition[0]}")
        if hasattr(model.net, "text_encoder"):
            with basic_utils.inference_mode(
                model.net.text_encoder, precision_amp=model.precision_amp_enc, device_type=model.device.type
            ):
                neg_condition = basic_utils.to(model.net.text_encoder.encode(neg_condition), **ctx)

    # Fix noise for all generated samples
    noise = torch.randn(
        [1, *config.model.input_shape],
        **ctx,
    )

    teacher_save_dir = save_dir / "teacher" if (has_teacher_sampling and has_student_sampling) else save_dir
    if has_teacher_sampling and has_student_sampling:
        teacher_save_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(pos_prompt_set):
        prompt_index = i
        logger.info(f"[{i+1}/{len(pos_prompt_set)}] Generating: {prompt[:80]}...")

        image_stem = Path(input_image_paths[i]).stem

        # Encode prompt
        condition = [prompt]
        if hasattr(model.net, "text_encoder"):
            with basic_utils.inference_mode(
                model.net.text_encoder, precision_amp=model.precision_amp_enc, device_type=model.device.type
            ):
                condition = basic_utils.to(model.net.text_encoder.encode(condition), **ctx)

        # VACE Wan mode: depth-to-video conditioning
        is_net_v2v = hasattr(model.net, "prepare_vid_conditioning")
        if source_video_paths is not None and i < len(source_video_paths) and is_net_v2v:
            depth_latent_path = depth_latent_paths[i] if depth_latent_paths is not None else None
            condition, neg_condition_sample = prepare_vacewan_condition(
                source_video_path=source_video_paths[i],
                depth_latent_path=depth_latent_path,
                model=model,
                latent_shape=config.model.input_shape,
                condition=condition,
                neg_condition=neg_condition,
                ctx=ctx,
            )
        else:
            neg_condition_sample = neg_condition

        # Image-to-video / Video2world mode
        i2v_tag = ""
        is_net_i2v = getattr(model.net, "is_i2v", False) or getattr(model.net, "is_video2world", False)
        if input_image_paths is not None and i < len(input_image_paths) and is_net_i2v:
            condition, neg_condition_sample, i2v_tag = prepare_i2v_condition(
                input_image_path=input_image_paths[i],
                model=model,
                vae=vae,
                latent_shape=config.model.input_shape,
                condition=condition,
                neg_condition=neg_condition,
                num_conditioning_frames=args.num_conditioning_frames,
                ctx=ctx,
            )

        # Student sampling
        if has_student_sampling:
            use_extrapolation = args.num_segments != 1 or args.overlap_frames != 0
            start_time = time.time()

            student_kwargs = {
                "condition": condition,
                "neg_condition": neg_condition_sample,
                "student_sample_steps": model.config.student_sample_steps,
                "student_sample_type": model.config.student_sample_type,
                "t_list": model.config.sample_t_cfg.t_list,
                "precision_amp": model.precision_amp_infer,
            }

            if use_extrapolation:
                if not hasattr(model, "generator_fn_extrapolation"):
                    raise RuntimeError("Extrapolation is only supported for causal autoregressive networks")
                if not hasattr(model.net, "vae"):
                    raise RuntimeError("VAE is required for extrapolation but was not initialized")
                student_kwargs["num_segments"] = args.num_segments
                student_kwargs["overlap_frames"] = args.overlap_frames
                video_student = model.generator_fn_extrapolation(student, noise, **student_kwargs)
            else:
                video_student = model.generator_fn(student, noise, **student_kwargs)

            sampling_time = time.time() - start_time
            logger.info(f"Student sampling time: {sampling_time:.2f}s")
            save_path = save_dir / f"{image_stem}-{args.video_index}.mp4"
            basic_utils.save_media(video_student, str(save_path), vae=vae, **save_video_kwargs)

        # Teacher sampling
        if has_teacher_sampling:
            start_time = time.time()
            teacher_kwargs = {
                "condition": condition,
                "neg_condition": neg_condition_sample,
                "num_steps": args.num_steps,
                "second_order": False,
                "precision_amp": model.precision_amp_infer,
                "fps": torch.full((noise.shape[0],), float(args.fps), device=noise.device),
            }
            if config.model.skip_layers is not None:
                teacher_kwargs["skip_layers"] = config.model.skip_layers

            video_teacher = model.sample(teacher, noise, **teacher_kwargs)
            sampling_time = time.time() - start_time
            logger.info(f"Teacher sampling time: {sampling_time:.2f}s")
            save_path = teacher_save_dir / f"{image_stem}-{args.video_index}.mp4"
            basic_utils.save_media(video_teacher, str(save_path), vae=vae, **save_video_kwargs)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video model inference (VBench-I2V output naming)")

    add_common_args(parser)

    parser.add_argument(
        "--save_as_gif",
        default=True,
        type=basic_utils.str2bool,
        help="Whether to save videos as GIF (True) or MP4 (False)",
    )
    parser.add_argument(
        "--fps",
        default=16,
        type=int,
        help="Frames per second for saved video and model temporal encoding (default: 16, matches Wan base_fps)",
    )
    parser.add_argument(
        "--save_high_quality",
        default=False,
        type=basic_utils.str2bool,
        help="Whether to save videos in high-quality (codec: libx265 vs libx264)",
    )
    parser.add_argument(
        "--prompt_file",
        default="scripts/inference/prompts/validation_aug_qwen_2_5_14b_seed42.txt",
        type=str,
        help="File containing prompts (one per line). Relative paths are resolved from script directory.",
    )
    parser.add_argument(
        "--neg_prompt_file",
        default="scripts/inference/prompts/negative_prompt.txt",
        type=str,
        help="The file containing the negative prompt to use for CFG.",
    )
    parser.add_argument(
        "--prompt_expand_model",
        type=str,
        help="If specified, perform prompt expansion using the specified Qwen model.",
        choices=["QwenVL2.5_3B", "QwenVL2.5_7B", "Qwen2.5_3B", "Qwen2.5_7B", "Qwen2.5_14B"],
    )
    parser.add_argument(
        "--prompt_expand_model_seed",
        type=int,
        help="Seed for prompt expansion.",
        default=0,
    )
    parser.add_argument(
        "--depth_latent_file",
        default=None,
        type=str,
        help="The file containing the depth latent paths to use for sampling.",
    )
    parser.add_argument(
        "--source_video_file",
        default="scripts/inference/prompts/source_video_paths.txt",
        type=str,
        help="The file containing the source video paths to use for sampling.",
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=1,
        help="Number of autoregressive segments to generate when using extrapolation (default: 1)",
    )
    parser.add_argument(
        "--overlap_frames",
        type=int,
        default=0,
        help="Number of latent frames to overlap between segments when extrapolating (default: 0)",
    )
    parser.add_argument(
        "--video_save_dir",
        type=str,
        help="Path to the video save directory (videos saved as <image_stem>-<idx>.mp4 directly inside).",
        default=None,
    )
    parser.add_argument(
        "--num_steps",
        default=50,
        type=int,
        help="Number of sampling steps for teacher (default: 50)",
    )
    parser.add_argument(
        "--input_image_file",
        type=str,
        default="scripts/inference/prompts/source_image_paths.txt",
        help="File containing paths to input images (one per line). Required — output filenames "
        "are derived from each image's basename (stem).",
    )
    parser.add_argument(
        "--num_conditioning_frames",
        type=int,
        default=1,
        help="Number of latent frames to condition on for I2V mode (default: 1).",
    )
    parser.add_argument(
        "--conditional_frame_timestep",
        type=float,
        default=0.0,
        help="Timestep value for conditioning frames in I2V mode.",
    )
    parser.add_argument(
        "--video_index",
        type=int,
        default=0,
        help="Sample index N for the VBench output filename '<image_stem>-<N>.mp4'. "
        "Re-run with different --video_index AND a different trainer.seed to populate "
        "0..4 (VBench averages over 5 seeds per prompt).",
    )

    args = parse_args(parser)
    config = setup(args, evaluation=True)
    main(args, config)

    clean_up()

# ----------------------------------------------------------------------------
