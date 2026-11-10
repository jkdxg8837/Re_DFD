# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import attrs
from omegaconf import DictConfig

from fastgen.utils import LazyCall as L
from fastgen.configs.config import (
    BaseModelConfig,
    BaseConfig,
)
from fastgen.configs.opt import BaseOptimizerConfig, BaseSchedulerConfig
from fastgen.methods import DMD2Model
from fastgen.configs.discriminator import Discriminator_EDM_CIFAR10_Config
from fastgen.configs.callbacks import (
    WANDB_CALLBACK,
    GradClip_CALLBACK,
    GPUStats_CALLBACK,
    TrainProfiler_CALLBACK,
    ParamCount_CALLBACK,
)


@attrs.define(slots=False)
class ModelConfig(BaseModelConfig):
    # optimizer and scheduler for the fake score net
    fake_score_optimizer: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseOptimizerConfig))
    fake_score_scheduler: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseSchedulerConfig))

    discriminator: DictConfig = attrs.field(factory=lambda: copy.deepcopy(Discriminator_EDM_CIFAR10_Config))
    # optimizer and scheduler for the discriminator
    discriminator_optimizer: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseOptimizerConfig))
    discriminator_scheduler: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseSchedulerConfig))

    # student update frequency
    student_update_freq: int = 5
    gen_real_replace_prob: float = 0.25
    forward_kl: bool = False ## whether to perform forward kl
    post_train: bool = False
    dmd_continue_remove_fake_score_discriminator: bool = False
    dmd_use_teacher_model_as_fake_score_estimate: bool = False

    # weight for the GAN loss in the student update phase
    gan_loss_weight_gen: float = 0.001

    # use the same t and noise to perturb the real data and fake data
    gan_use_same_t_noise: bool = False

    # perform dsm loss in a space specified by fake_score_pred_type (None means use original net_pred_type)
    fake_score_pred_type: str | None = None

    # R1 regularization weight (0 means no R1 reg, recommended value when using R1 reg: 100-1000)
    gan_r1_reg_weight: float = 0.0
    # R1 regularization noise scale (it only takes effect when gan_r1_reg_weight > 0)
    gan_r1_reg_alpha: float = 0.1

    # DFD experiments (spec §3 E2s): if non-empty, the multi-step student input is built from data[<key>] (a second
    # video stream, e.g. a paired ViPE clip) instead of data["real"]; data["real"] still feeds the teacher-score
    # replacement. "" (default) keeps the original behavior.
    dfd_student_input_key: str = ""
    # DFD experiments (spec §4 F2): every N-th student update, log reg_mean_norm/reg_var (via count sketches of the
    # teacher-score difference), delta_sq and lipschitz_hat to dfd_diagnostics_dir. 0 (default) = off.
    dfd_diagnostics_every: int = 0
    dfd_diagnostics_dir: str = ""
    dfd_diagnostics_sketch_dim: int = 65536
    # DFD experiments (spec §3 E3): on replaced batches, evaluate the teacher's CFG unconditional branch at the noised
    # generated point instead of the noised real point. False (default) keeps the original behavior.
    dfd_cfg_uncond_on_gen: bool = False
    # DFD experiments (spec §3 E4): if non-empty, eps_student is taken from data[<key>] (teacher-ODE inverted noise
    # of the real sample, e.g. "z_inv") instead of torch.randn. "" (default) keeps the original behavior.
    dfd_inverted_noise_key: str = ""
    # Phase B (task2.md B0.1): per sample, teacher-score samples with t >= cutoff use w = 1 (no guidance), the others
    # guidance_scale; applies wherever _apply_classifier_free_guidance is called. None (default) = original behavior.
    guidance_high_t_cutoff: float | None = None
    # Phase B (task2.md B0.2): no-grad logging of r = fake_score_x0 - teacher_x0 per timestep bin to
    # <dfd_residual_log_dir>/residuals.jsonl. "" (default) = off. dfd_residual_mu_dir_path: tensor with mu_R - mu_S
    # (latent shape, Phase 0) for the projection. dfd_residual_accum_rounds must equal trainer.grad_accum_rounds:
    # the batch statistics are taken over the global batch (all ranks x all accumulation rounds of one iteration).
    dfd_residual_log_dir: str = ""
    dfd_residual_mu_dir_path: str = ""
    dfd_residual_accum_rounds: int = 1


@attrs.define(slots=False)
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = L(DMD2Model)(
        config=None,
    )


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **GPUStats_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
            **WANDB_CALLBACK,
        }
    )

    config.dataloader_train.batch_size = 256
    config.model.discriminator_scheduler.warm_up_steps = [0]
    config.model.fake_score_scheduler.warm_up_steps = [0]
    config.model.net_scheduler.warm_up_steps = [0]
    config.model.sample_t_cfg.time_dist_type = "polynomial"

    return config
