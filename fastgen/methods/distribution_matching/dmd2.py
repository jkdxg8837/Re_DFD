# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import Dict, Any, TYPE_CHECKING, Callable, Optional

import torch
import torch.nn.functional as F

from fastgen.configs.opt import get_scheduler
from fastgen.utils import instantiate
from fastgen.methods import FastGenModel
from fastgen.methods.common_loss import (
    denoising_score_matching_loss,
    variational_score_distillation_loss,
    gan_loss_generator,
    gan_loss_discriminator,
)
import fastgen.utils.logging_utils as logger
from fastgen.utils.distributed import synchronize, is_rank0
from fastgen.utils.basic_utils import convert_cfg_to_dict


if TYPE_CHECKING:
    from fastgen.configs.methods.config_dmd2 import ModelConfig


class DMD2Model(FastGenModel):
    def __init__(self, config: ModelConfig):
        """

        Args:
            config (ModelConfig): The configuration for the DMD model
        """
        super().__init__(config)
        self.config = config

    def build_model(self):
        super().build_model()
        self.build_teacher()
        self.load_student_weights_and_ema()

        # instantiate the fake_score and load weights from teacher
        logger.info("Instantiating the fake_score")
        with self._get_meta_init_context():
            self.fake_score = instantiate(self.teacher_config)
        model_path = self.config.pretrained_model_path
        if model_path is not None and len(model_path) > 0:
            if (not self.config.fsdp_meta_init) or is_rank0():
                # Only rank 0 loads weights if using meta initialization
                self.fake_score.load_state_dict(self.teacher.state_dict())
            synchronize()

        if self.config.gan_loss_weight_gen > 0:
            logger.info(f"gan_loss_weight_gen: {self.config.gan_loss_weight_gen}")
            # instantiate the discriminator in DMD2
            logger.info("Instantiating the discriminator")
            if getattr(self.config.discriminator, "disc_type", None) is not None:
                logger.info(f"Discriminator type: {self.config.discriminator.disc_type}")
            # TODO: Discriminators do not yet support meta initialization
            self.discriminator = instantiate(self.config.discriminator)
            synchronize()
        torch.cuda.empty_cache()

    def _setup_grad_requirements(self, iteration: int) -> None:
        if iteration % self.config.student_update_freq == 0:
            # update the student
            self.fake_score.eval().requires_grad_(False)
            if self.config.gan_loss_weight_gen > 0:
                self.discriminator.eval().requires_grad_(False)
        else:
            # update the fake_score and discriminator
            self.fake_score.train().requires_grad_(True)
            if self.config.gan_loss_weight_gen > 0:
                self.discriminator.train().requires_grad_(True)

    def _generate_noise_and_time(
        self, real_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random noises and time step

        Args:
            batch_size: Batch size
            real_data: Real data tensor for dtype/device reference

        Returns:
            rand_z_max: Random noise used by the student
            t_max: Time step used by the student
            t: Time step
            eps: Random noise used by a forward process
        """
        batch_size = real_data.shape[0]

        eps_student = torch.randn(batch_size, *self.input_shape, device=self.device, dtype=real_data.dtype)
        if self.config.student_sample_steps == 1:
            # perform single-step distillation
            # input noise to student (sigma * eps)
            t_student = torch.full(
                (batch_size,),
                self.net.noise_scheduler.max_t,
                device=self.device,
                dtype=self.net.noise_scheduler.t_precision,
            )
            input_student = self.net.noise_scheduler.latents(noise=eps_student)
        else:
            # perform multiple-step distillation
            # Add noise to real image data (for multistep generation)
            t_student = self.net.noise_scheduler.sample_from_t_list(
                batch_size,
                sample_steps=self.config.student_sample_steps,
                t_list=self.config.sample_t_cfg.t_list,
                device=self.device,
            )
            input_student = self.net.noise_scheduler.forward_process(real_data, eps_student, t_student)

        t = self.net.noise_scheduler.sample_t(
            batch_size, **convert_cfg_to_dict(self.config.sample_t_cfg), device=self.device
        )
        eps = torch.randn_like(real_data, device=self.device, dtype=real_data.dtype)
        return input_student, t_student, t, eps

    def _compute_teacher_prediction_gan_loss(
        self, perturbed_data: torch.Tensor, perturbed_data_teacher_score:torch.Tensor,t: torch.Tensor, condition: Optional[Any] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher prediction and optionally GAN loss for generator.

        Args:
            perturbed_data: Perturbed data tensor
            t: Time steps
            condition: Conditioning information

        Returns:
            tuple of (teacher_x0, fake_feat or None, gan_loss_gen)
        """
        
        if self.config.dmd_continue_remove_fake_score_discriminator == False:
            if self.config.gan_loss_weight_gen > 0:
                teacher_x0_teacher_score, fake_feat_one_is_true = self.teacher(
                    perturbed_data_teacher_score,
                    t,
                    condition=condition,
                    feature_indices=self.discriminator.feature_indices,
                    fwd_pred_type="x0",
                )
                teacher_x0, fake_feat = self.teacher(
                    perturbed_data,
                    t,
                    condition=condition,
                    feature_indices=self.discriminator.feature_indices,
                    fwd_pred_type="x0",
                )
                # Compute the GAN loss for the generator
                gan_loss_gen = gan_loss_generator(self.discriminator(fake_feat))
            else:
                teacher_x0_teacher_score = self.teacher(
                    perturbed_data_teacher_score,
                    t,
                    condition=condition,
                    fwd_pred_type="x0",
                )
                gan_loss_gen = torch.tensor(0.0, device=self.device, dtype=teacher_x0_teacher_score.dtype)
        else:
            feature_indices = {29, 22, 15}
            teacher_x0_teacher_score, fake_feat_one_is_true = self.teacher(
                    perturbed_data_teacher_score,
                    t,
                    condition=condition,
                    feature_indices=feature_indices,
                    fwd_pred_type="x0",
                )
            gan_loss_gen = torch.tensor(0.0, device=self.device, dtype=teacher_x0_teacher_score.dtype)


        return teacher_x0_teacher_score.detach(), gan_loss_gen

    def _apply_classifier_free_guidance(
        self,
        perturbed_data: torch.Tensor,
        t: torch.Tensor,
        teacher_x0: torch.Tensor,
        neg_condition: Optional[Any] = None,
    ) -> torch.Tensor:
        """Apply classifier-free guidance to teacher predictions.

        Args:
            perturbed_data: Perturbed data
            t: Time step
            teacher_x0: Original teacher x0 prediction
            neg_condition: Negative conditioning for CFG

        Returns:
            CFG-adjusted teacher_x0
        """
        assert self.config.guidance_scale is not None, "guidance_scale must be provided"
        # classifier-free guidance
        with torch.no_grad():
            kwargs = {"condition": neg_condition, "fwd_pred_type": "x0"}
            if self.config.skip_layers is not None:
                kwargs["skip_layers"] = self.config.skip_layers
            teacher_x0_neg = self.teacher(perturbed_data, t, **kwargs)

        cutoff = getattr(self.config, "guidance_high_t_cutoff", None)
        if cutoff is None:
            teacher_x0 = teacher_x0 + (self.config.guidance_scale - 1) * (teacher_x0 - teacher_x0_neg)
        else:
            # Phase B (task2.md B0.1): per-sample guidance, w = 1 for t >= cutoff, guidance_scale otherwise
            w = torch.where(t >= cutoff, 1.0, float(self.config.guidance_scale)).to(teacher_x0.dtype)
            w = w.view(-1, *([1] * (teacher_x0.ndim - 1)))
            teacher_x0 = teacher_x0 + (w - 1) * (teacher_x0 - teacher_x0_neg)
        return teacher_x0

    def _student_update_step(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        data: Dict[str, Any],
        real_data: torch.Tensor,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Perform student model update step.

        Args:
            input_student: Input tensor to student network
            t_student: Input time to student network
            t: Time step
            eps: Noise tensor
            data: Original data batch
            condition: Conditioning information
            neg_condition: Negative conditioning

        Returns:
            tuple of (loss_map, outputs)
        """
        # Generate data from student
        gen_data = self.gen_data_from_net(input_student, t_student, condition=condition)
        gen_data_teacher_score = gen_data
        forward_kl_ensemble=False ## TODO: here do the minus signal change
       
        if self.config.gen_real_replace_prob > 0:
            if (torch.rand((), device=gen_data.device) < self.config.gen_real_replace_prob) and (self.config.post_train==False):
                gen_idx = torch.randint(0, gen_data.shape[0], (1,), device=gen_data.device).item()
                real_idx = torch.randint(0, real_data.shape[0], (1,), device=real_data.device).item()
                gen_data_teacher_score = gen_data.clone()
                real_sample = real_data[real_idx].to(device=gen_data.device, dtype=gen_data.dtype)
                gen_data_teacher_score[gen_idx] = gen_data_teacher_score[gen_idx] + (real_sample - gen_data[gen_idx]).detach()
                forward_kl_ensemble=True 
            elif (torch.rand((), device=gen_data.device) < self.config.gen_real_replace_prob) and (self.config.post_train==True):
                gen_idx = torch.randint(0, gen_data.shape[0], (1,), device=gen_data.device).item()
                real_idx = torch.randint(0, real_data.shape[0], (1,), device=real_data.device).item()
                gen_data_teacher_score = gen_data.clone()
                real_sample = real_data.to(device=gen_data.device, dtype=gen_data.dtype)
                gen_data_teacher_score = gen_data_teacher_score + (real_sample - gen_data).detach() ## fully use our method
                forward_kl_ensemble=True


        # else:
        #     ### here we totally remove the fake score and discriminator and do a optimial diffsion distribution matching
        #     gen_data_teacher_score = gen_data.clone()
        #     real_sample = real_data.to(device=gen_data.device, dtype=gen_data.dtype)
        #     gen_data_teacher_score = gen_data_teacher_score + (real_sample - gen_data).detach()
        perturbed_data = self.net.noise_scheduler.forward_process(gen_data, eps, t)
        perturbed_data_teacher_score = self.net.noise_scheduler.forward_process(gen_data_teacher_score, eps, t)

        # Compute the fake score with x0-prediction
        if not self.config.dmd_use_teacher_model_as_fake_score_estimate:
            with torch.no_grad():
                if self.config.forward_kl ==True:
                    fake_score_x0 = self.fake_score(perturbed_data_teacher_score, t, condition=condition, fwd_pred_type="x0")
                else:
                    fake_score_x0 = self.fake_score(perturbed_data, t, condition=condition, fwd_pred_type="x0")
        else:
            with torch.no_grad():
                fake_score_x0 = self.teacher(
                        perturbed_data,
                        t,
                        condition=condition,
                        fwd_pred_type="x0",
                    )

        # Compute the teacher x0-prediction and gan loss for generator
        assert (
            perturbed_data.dtype == data["real"].dtype == input_student.dtype
        ), f"perturbed_data.dtype: {perturbed_data.dtype}, data['real'].dtype: {data['real'].dtype}, input_student.dtype: {input_student.dtype}"
        assert (
            t.dtype == t_student.dtype == self.net.noise_scheduler.t_precision
        ), f"t.dtype: {t.dtype}, t_student.dtype: {t_student.dtype}, self.net.noise_scheduler.t_precision: {self.net.noise_scheduler.t_precision}"
        teacher_x0, gan_loss_gen = self._compute_teacher_prediction_gan_loss(perturbed_data, perturbed_data_teacher_score, t, condition=condition)

        # Apply classifier-free guidance if needed
        if self.config.guidance_scale is not None:
            if self.config.dfd_cfg_uncond_on_gen and forward_kl_ensemble:
                # DFD experiments (spec §3 E3, default off): cond branch at the real point (teacher_x0 above),
                # uncond branch at the generated point
                teacher_x0 = self._apply_classifier_free_guidance(
                    perturbed_data, t, teacher_x0, neg_condition=neg_condition
                )
            else:
                teacher_x0 = self._apply_classifier_free_guidance(
                    perturbed_data_teacher_score, t, teacher_x0, neg_condition=neg_condition
                )

        # Phase B (default off): no-grad residual logging (task2.md B0.2), no effect on the loss
        if self.config.dfd_residual_log_dir:
            self._phaseab_log_residuals(gen_data, teacher_x0, fake_score_x0, t, t_student, forward_kl_ensemble)

        # Compute the VSD loss
        vsd_loss = variational_score_distillation_loss(gen_data, teacher_x0, fake_score_x0, forward_kl_ensemble)

        # Compute the final loss
        loss = vsd_loss + self.config.gan_loss_weight_gen * gan_loss_gen

        # Build output dictionaries
        loss_map = {
            "total_loss": loss,
            "vsd_loss": vsd_loss,
            "gan_loss_gen": gan_loss_gen,
        }
        # DFD experiments (default off): no-grad F2 diagnostics (spec §4), no effect on the loss
        if self.config.dfd_diagnostics_every > 0:
            self._dfd_log_diagnostics(
                gen_data, real_data, eps, t, teacher_x0, forward_kl_ensemble, condition=condition, neg_condition=neg_condition
            )
        outputs = self._get_outputs(gen_data, input_student, condition=condition)

        return loss_map, outputs

    # ------------------------------------------------------------------------------------------------------------
    # DFD experiments: helpers behind default-off flags (dfd_student_input_key, dfd_diagnostics_every)
    # ------------------------------------------------------------------------------------------------------------
    def _dfd_student_input(
        self, data: Dict[str, Any], input_student: torch.Tensor, t_student: torch.Tensor, real_data: torch.Tensor
    ) -> torch.Tensor:
        """Multi-step student input forward_process(data[key], eps', t_student) instead of noising data["real"].

        eps' is a fresh draw (the original eps_student is not returned by _generate_noise_and_time); t_student is kept.
        Single-step students take pure noise, so their input is returned unchanged.
        """
        if self.config.student_sample_steps == 1:
            return input_student
        x = data[self.config.dfd_student_input_key]
        if hasattr(self.net, "vae") and x.shape[1] != self.input_shape[0]:
            # raw video [B, C, T, H, W] -> latent, as trainer.preprocess_data does for data["real"]
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type, dtype=self.precision_amp_enc, enabled=self.precision_amp_enc is not None
            ):
                x = self.net.vae.encode(x)
        x = x.to(device=real_data.device, dtype=real_data.dtype)
        assert x.shape == real_data.shape, f"student input {tuple(x.shape)} vs real {tuple(real_data.shape)}"
        eps_student = torch.randn_like(real_data)
        return self.net.noise_scheduler.forward_process(x, eps_student, t_student)

    def _dfd_inverted_input(
        self, data: Dict[str, Any], input_student: torch.Tensor, t_student: torch.Tensor, real_data: torch.Tensor
    ) -> torch.Tensor:
        """Student input with eps_student = data[key] (teacher-ODE inversion of the real sample) instead of randn.

        Multi-step: forward_process(real_data, z_inv, t_student) (t_student unchanged); single-step: latents(z_inv).
        """
        z = data[self.config.dfd_inverted_noise_key].to(device=real_data.device, dtype=real_data.dtype)
        assert z.shape == real_data.shape, f"z_inv {tuple(z.shape)} vs real {tuple(real_data.shape)}"
        if self.config.student_sample_steps == 1:
            return self.net.noise_scheduler.latents(noise=z)
        return self.net.noise_scheduler.forward_process(real_data, z, t_student)

    def _dfd_guided_teacher_x0(self, x_t, t, condition=None, neg_condition=None) -> torch.Tensor:
        """Teacher x0 at x_t exactly as the student update computes it (GAN off): cond forward + CFG."""
        x0 = self.teacher(x_t, t, condition=condition, fwd_pred_type="x0").detach()
        if self.config.guidance_scale is not None:
            x0 = self._apply_classifier_free_guidance(x_t, t, x0, neg_condition=neg_condition)
        return x0

    def _dfd_sketch(self, v: torch.Tensor) -> torch.Tensor:
        """Count sketch [B, N] -> [B, D] with a fixed (seed 0) hash, so <S a, S b> is unbiased for <a, b>.

        Uses its own CPU generator: the global RNG stream (and hence training) is untouched.
        """
        D = self.config.dfd_diagnostics_sketch_dim
        sk = getattr(self, "_dfd_sk", None)
        if sk is None or sk[0].numel() != v.shape[1]:
            g = torch.Generator().manual_seed(0)
            idx = torch.randint(0, D, (v.shape[1],), generator=g)
            sign = torch.randint(0, 2, (v.shape[1],), generator=g).float() * 2 - 1
            self._dfd_sk = sk = (idx.to(v.device), sign.to(v.device))
        out = torch.zeros(v.shape[0], D, device=v.device, dtype=torch.float32)
        out.index_add_(1, sk[0], v.float() * sk[1])
        return out

    def _phaseab_log_residuals(self, gen_data, teacher_x0, fake_score_x0, t, t_student, replaced) -> None:
        """Residual r = fake_score_x0 - teacher_x0 (x0 space, as in the VSD loss; teacher_x0 after CFG) (task2.md B0.2).

        Logged at student updates k <= 10 and every 5th after (k = iteration // student_update_freq). "Batch" = the
        global batch of one iteration: every rank accumulates its micro-batches for dfd_residual_accum_rounds calls,
        then the sums are all-reduced and rank 0 appends one row to residuals.jsonl. Samples are binned two ways by
        the nearest of the student t_list values: by the student's input time t_student (which student step made
        G) and by the teacher-score time t (continuous; nearest t_list value). Each bin is split by whether the
        micro-batch had the DFD replacement. Per-sample rows go to residual_samples_rank<r>.jsonl. No RNG is used.
        """
        import json
        import os

        k = self._dfd_iteration // self.config.student_update_freq
        if not (k <= 10 or k % 5 == 0):
            return
        tl = self.config.sample_t_cfg.t_list
        if tl is None:  # same default as noise_scheduler.sample_from_t_list
            tl = self.net.noise_scheduler.get_t_list(sample_steps=self.config.student_sample_steps).tolist()
        tl = [float(x) for x in list(tl)[: self.config.student_sample_steps]]
        nb = len(tl)
        with torch.no_grad():
            r = (fake_score_x0.detach().float() - teacher_x0.detach().float()).flatten(1)  # [B, N]
            dam = (gen_data.detach().float() - teacher_x0.detach().float()).abs().flatten(1).mean(1)  # VSD normaliser
            ms = r.pow(2).mean(1)
            tk = torch.tensor(tl, device=r.device, dtype=torch.float64)
            bins = torch.stack([
                (t_student.double()[:, None] - tk[None]).abs().argmin(1),
                (t.double()[:, None] - tk[None]).abs().argmin(1),
            ])  # [2 groupings, B]
            rep = int(bool(replaced))
            acc = getattr(self, "_phaseab_res", None)
            if acc is None or acc["iteration"] != self._dfd_iteration:
                if acc is not None and acc["calls"] > 0:
                    logger.warning(f"residual log: iteration {acc['iteration']} flushed incomplete ({acc['calls']} calls)")
                    self._phaseab_flush_residuals(acc, tl, complete=False)
                acc = self._phaseab_res = {
                    "iteration": self._dfd_iteration, "calls": 0,
                    "vec": torch.zeros(2, nb, 2, r.shape[1], device=r.device, dtype=torch.float32),
                    "sc": torch.zeros(2, nb, 2, 3, device=r.device, dtype=torch.float64),  # n, sum ms, sum dam
                }
            for b in range(r.shape[0]):
                for g in range(2):
                    i = int(bins[g, b])
                    acc["vec"][g, i, rep] += r[b]
                    acc["sc"][g, i, rep] += torch.stack([torch.ones((), device=r.device, dtype=torch.float64),
                                                          ms[b].double(), dam[b].double()])
            acc["calls"] += 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        out_dir = self.config.dfd_residual_log_dir
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"residual_samples_rank{rank}.jsonl"), "a") as f:
            for b in range(r.shape[0]):
                f.write(json.dumps({
                    "iteration": int(self._dfd_iteration), "student_step": int(k), "rank": rank,
                    "call": int(acc["calls"]), "b": b, "t": float(t[b]), "t_student": float(t_student[b]),
                    "replaced": bool(replaced), "mean_r_sq": float(ms[b]), "diff_abs_mean": float(dam[b]),
                }) + "\n")
        if acc["calls"] >= max(1, self.config.dfd_residual_accum_rounds):
            self._phaseab_flush_residuals(acc, tl, complete=True)
            self._phaseab_res = None

    def _phaseab_flush_residuals(self, acc, tl, complete: bool) -> None:
        """All-reduce the residual sums of one iteration and write one row (rank 0) to residuals.jsonl."""
        import json
        import os

        vec, sc = acc["vec"], acc["sc"]
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(vec)
            torch.distributed.all_reduce(sc)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank != 0:
            return
        u = getattr(self, "_phaseab_u", None)
        if u is None and self.config.dfd_residual_mu_dir_path:
            p = self.config.dfd_residual_mu_dir_path
            if p.endswith(".npy"):
                import numpy as np

                d = torch.from_numpy(np.load(p))
            else:
                d = torch.load(p, map_location="cpu")
                if isinstance(d, dict):
                    d = d.get("mu_dir", d.get("u"))
            d = d.flatten().float().to(vec.device)
            assert d.numel() == vec.shape[-1], f"mu_dir has {d.numel()} elements, residual {vec.shape[-1]}"
            u = self._phaseab_u = d / d.norm()

        def stats(v, s):  # v: [N] sum of r, s: [3] = n, sum mean_r_sq, sum diff_abs_mean
            n = int(s[0])
            if n == 0:
                return {"n": 0}
            m = v / n
            mean_r_sq = float(s[1] / n)
            out = {"n": n, "mean_r_sq": mean_r_sq,
                   "common_mode_ratio": float(m.pow(2).mean()) / mean_r_sq if mean_r_sq > 0 else None,
                   "diff_abs_mean": float(s[2] / n)}
            if u is not None:
                proj = float(m @ u)
                out["proj_mu_dir"] = proj  # <mean_batch(r), unit(mu_R - mu_S)>
                out["cos_mu_dir"] = proj / max(float(m.norm()), 1e-30)
            return out

        row = {"iteration": int(acc["iteration"]), "student_step": int(acc["iteration"] // self.config.student_update_freq),
               "complete": complete, "calls_per_rank": int(acc["calls"]), "t_list": tl}
        for g, name in enumerate(("by_t_student", "by_t_score")):
            grp = {}
            for i in list(range(len(tl))) + ["all"]:
                vi = vec[g].sum(0) if i == "all" else vec[g, i]  # [2 (rep), N]
                si = sc[g].sum(0) if i == "all" else sc[g, i]  # [2, 3]
                key = "all" if i == "all" else f"{tl[i]:g}"
                grp[key] = {"all": stats(vi.sum(0), si.sum(0)), "replaced": stats(vi[1], si[1]),
                            "not_replaced": stats(vi[0], si[0])}
            row[name] = grp
        out_dir = self.config.dfd_residual_log_dir
        with open(os.path.join(out_dir, "residuals.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")

    def _dfd_log_diagnostics(
        self, gen_data, real_data, eps, t, teacher_x0, replaced, condition=None, neg_condition=None
    ) -> None:
        """F2 diagnostics 1-3 (spec §4), logged every dfd_diagnostics_every-th student update, per sample.

        Score from an x0 prediction: s(x_t) = (alpha_t x0_hat - x_t) / sigma_t^2 (x_t = alpha_t x0 + sigma_t eps).
        Delta = s_teacher(x_t^real) - s_teacher(x_t^gen), both with the CFG-guided teacher, same eps and t (as in
        the DFD update). Per sample: delta_sq = mean (x_real - G)^2 (latent), score_diff_sq = ||Delta||^2,
        lipschitz_hat = ||Delta|| / ||x_t^real - x_t^gen||, plus a count sketch of Delta; reg_mean_norm / reg_var are
        computed offline from the sketches (unbiased pairwise estimator) per t bin.
        """
        k = self._dfd_iteration // self.config.student_update_freq  # student-update index (1, 2, ...)
        if (k - 1) % self.config.dfd_diagnostics_every != 0:
            return
        import json
        import os

        ns = self.net.noise_scheduler
        with torch.no_grad():
            G = gen_data.detach()
            xt_real = ns.forward_process(real_data, eps, t)
            xt_gen = ns.forward_process(G, eps, t)
            if self.config.post_train and not self.config.dfd_cfg_uncond_on_gen:
                # teacher_x0 (used by the loss) sits at the real point iff this batch was replaced (whole batch)
                x0_real = teacher_x0 if replaced else self._dfd_guided_teacher_x0(xt_real, t, condition, neg_condition)
                x0_gen = self._dfd_guided_teacher_x0(xt_gen, t, condition, neg_condition) if replaced else teacher_x0
            else:
                # partial replacement, or E3's mixed CFG target: recompute both points with the standard CFG on
                # every rank (same number of FSDP forward calls on all ranks)
                x0_real = self._dfd_guided_teacher_x0(xt_real, t, condition, neg_condition)
                x0_gen = self._dfd_guided_teacher_x0(xt_gen, t, condition, neg_condition)
            shape = (-1,) + (1,) * (G.ndim - 1)
            a = ns.alpha(t).view(shape).float()
            s2 = ns.sigma(t).view(shape).float() ** 2
            delta = ((a * x0_real.float() - xt_real.float()) - (a * x0_gen.float() - xt_gen.float())) / s2
            delta = delta.flatten(1)
            dxt = (xt_real.float() - xt_gen.float()).flatten(1)
            dx = (real_data.float() - G.float()).flatten(1)
            dx0 = (x0_real.float() - x0_gen.float()).flatten(1)
            sketch = self._dfd_sketch(delta)

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        out_dir = self.config.dfd_diagnostics_dir or "dfd_diag"
        os.makedirs(out_dir, exist_ok=True)
        self._dfd_diag_calls = getattr(self, "_dfd_diag_calls", 0) + 1
        tag = f"it{self._dfd_iteration:06d}_r{rank}_c{self._dfd_diag_calls:05d}"
        torch.save({"sketch": sketch.cpu(), "t": t.float().cpu()}, os.path.join(out_dir, f"sketch_{tag}.pt"))
        with open(os.path.join(out_dir, f"rank{rank}.jsonl"), "a") as f:
            for b in range(G.shape[0]):
                f.write(json.dumps({
                    "iteration": int(self._dfd_iteration), "student_step": int(k), "rank": rank, "tag": tag, "b": b,
                    "t": float(t[b]), "alpha": float(a.flatten()[b]), "sigma": float(s2.flatten()[b].sqrt()),
                    "replaced": bool(replaced), "numel": int(delta.shape[1]),
                    "delta_sq": float(dx[b].pow(2).mean()),
                    "score_diff_sq": float(delta[b].pow(2).sum()),
                    "xt_diff_norm": float(dxt[b].norm()),
                    "lipschitz_hat": float(delta[b].norm() / dxt[b].norm().clamp_min(1e-12)),
                    "teacher_x0_diff_sq": float(dx0[b].pow(2).mean()),
                }) + "\n")

    def _compute_real_feat(
        self, real_data: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, condition: Optional[Any] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute discriminator features for both real and fake data.

        Args:
            real_data: Real data tensor
            t: Time step
            eps: Noise tensor
            condition: Conditioning information

        Returns:
            tuple of (real_feat, t_real)
        """
        # decide whether to use the same t and noise for real and fake data
        if self.config.gan_use_same_t_noise:
            t_real = t
            eps_real = eps
        else:
            t_real = self.net.noise_scheduler.sample_t(
                real_data.shape[0],
                **convert_cfg_to_dict(self.config.sample_t_cfg),
                device=self.device,
            )
            eps_real = torch.randn_like(real_data)
        # Perturb the real data according to the given forward process
        perturbed_real = self.net.noise_scheduler.forward_process(real_data, eps_real, t_real)
        real_feat = self.teacher(
            perturbed_real,
            t_real,
            condition=condition,
            return_features_early=True,
            feature_indices=self.discriminator.feature_indices,
        )

        return real_feat, t_real

    def _compute_r1_regularization(
        self,
        real_feat_logit: torch.Tensor,
        real_data: torch.Tensor,
        t_real: torch.Tensor,
        condition: Optional[Any] = None,
    ) -> torch.Tensor:
        """Compute R1 regularization loss for discriminator.

        Args:
            real_feat_logit: Real feature logits
            real_data: Real data tensor
            t_real: Time step for real data
            condition: Conditioning information

        Returns:
            R1 regularization loss
        """
        perturbed_real_alpha = real_data.add(self.config.gan_r1_reg_alpha * torch.randn_like(real_data))
        with torch.no_grad():
            real_feat_alpha = self.teacher(
                perturbed_real_alpha,
                t_real,
                condition=condition,
                return_features_early=True,
                feature_indices=self.discriminator.feature_indices,
            )
        real_feat_alpha_logit = self.discriminator(real_feat_alpha)
        gan_loss_ar1 = F.mse_loss(real_feat_logit, real_feat_alpha_logit, reduction="mean")

        return gan_loss_ar1

    def _fake_score_discriminator_update_step(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        real_data: torch.Tensor,
        condition: Optional[Any] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Perform fake score and discriminator update step.

        Args:
            input_student: Input tensor to student network
            t_student: Input time to student network
            t: Time steps
            eps: Noise tensor
            condition: Conditioning information
            real_data: Real data tensor

        Returns:
            tuple of (loss_map, outputs)
        """
        # Generate data and compute fake score loss
        with torch.no_grad():
            gen_data = self.gen_data_from_net(input_student, t_student, condition=condition)
            x_t_sg = self.net.noise_scheduler.forward_process(gen_data, eps, t)

        # The fake score matches the teacher, but we want to do SDS in x0 space
        fake_score_pred_type = self.config.fake_score_pred_type or self.teacher.net_pred_type
        assert (
            x_t_sg.dtype == real_data.dtype == input_student.dtype
        ), f"x_t_sg.dtype: {x_t_sg.dtype}, real_data.dtype: {real_data.dtype}, input_student.dtype: {input_student.dtype}"
        assert (
            t.dtype == t_student.dtype == self.net.noise_scheduler.t_precision
        ), f"t.dtype: {t.dtype}, t_student.dtype: {t_student.dtype}, self.net.noise_scheduler.t_precision: {self.net.noise_scheduler.t_precision}"
        fake_score_pred = self.fake_score(x_t_sg, t, condition=condition, fwd_pred_type=fake_score_pred_type)
        loss_fakescore = denoising_score_matching_loss(
            fake_score_pred_type,
            net_pred=fake_score_pred,
            noise_scheduler=self.net.noise_scheduler,
            x0=gen_data,
            eps=eps,
            t=t,
        )

        gan_loss_disc = torch.zeros_like(loss_fakescore)
        gan_loss_ar1 = torch.zeros_like(loss_fakescore)
        if self.config.gan_loss_weight_gen > 0:
            # Compute the GAN loss for the discriminator
            with torch.no_grad():
                fake_feat = self.teacher(
                    x_t_sg,
                    t,
                    condition=condition,
                    return_features_early=True,
                    feature_indices=self.discriminator.feature_indices,
                )

                real_feat, t_real = self._compute_real_feat(real_data=real_data, t=t, eps=eps, condition=condition)

            real_feat_logit = self.discriminator(real_feat)
            gan_loss_disc = gan_loss_discriminator(real_feat_logit, self.discriminator(fake_feat))

            # Use approximate R1 regularization in the APT paper to regularize the discriminator head
            if self.config.gan_r1_reg_weight > 0:
                gan_loss_ar1 = self._compute_r1_regularization(real_feat_logit, real_data, t_real, condition=condition)

        loss = loss_fakescore + gan_loss_disc + self.config.gan_r1_reg_weight * gan_loss_ar1

        loss_map = {
            "total_loss": loss,
            "fake_score_loss": loss_fakescore,
            "gan_loss_disc": gan_loss_disc,
        }
        if self.config.gan_loss_weight_gen > 0 and self.config.gan_r1_reg_weight > 0:
            loss_map.update({"gan_loss_ar1": gan_loss_ar1})
        outputs = self._get_outputs(gen_data, input_student, condition=condition)

        return loss_map, outputs

    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: torch.Tensor = None,
        condition: Any = None,
    ) -> Dict[str, torch.Tensor | Callable]:
        if self.config.student_sample_steps == 1:
            assert input_student is not None, "input_student must be provided"
            noise = input_student / self.net.noise_scheduler.max_sigma
            return {"gen_rand": gen_data, "input_rand": noise}
        else:
            noise = torch.randn_like(gen_data, dtype=self.precision)
            gen_rand_func = partial(
                self.generator_fn,
                net=self.net_inference,
                noise=noise,
                condition=condition,
                student_sample_steps=self.config.student_sample_steps,
                student_sample_type=self.config.student_sample_type,
                t_list=self.config.sample_t_cfg.t_list,
                precision_amp=self.precision_amp_infer,
            )
            return {"gen_rand": gen_rand_func, "input_rand": noise, "gen_rand_train": gen_data}

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for distribution matching distillation (DMD)

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output

        """
        # Prepare training data and conditions
        real_data, condition, neg_condition = self._prepare_training_data(data)

        # Set up gradient requirements based on training phase
        self._setup_grad_requirements(iteration)

        # Generate noise and time steps
        input_student, t_student, t, eps = self._generate_noise_and_time(real_data)
        # DFD experiments (default off): student input from a second video stream (spec §3 E2s)
        if self.config.dfd_student_input_key:
            input_student = self._dfd_student_input(data, input_student, t_student, real_data)
        # DFD experiments (default off): eps_student from teacher-ODE inversion of the real sample (spec §3 E4)
        if self.config.dfd_inverted_noise_key:
            assert not self.config.dfd_student_input_key, "E4 inverted noise is defined for the E2 arm only"
            input_student = self._dfd_inverted_input(data, input_student, t_student, real_data)
        self._dfd_iteration = iteration

        # print(self.discriminator.feature_indices)
        # assert 1==0

        # Choose between student update or fake_score/discriminator update
        if not self.config.dmd_use_teacher_model_as_fake_score_estimate:
            if iteration % self.config.student_update_freq == 0:
                # print("student update")
                return self._student_update_step(
                    input_student, t_student, t, eps, data, real_data,condition=condition, neg_condition=neg_condition
                )
            else:
                # print("gan update")
                return self._fake_score_discriminator_update_step(
                    input_student, t_student, t, eps, real_data, condition=condition
                )
        else:
            # if we use teacher model as fake score estimate, then we only update the student and do not have a separate fake score/discriminator update step
            return self._student_update_step(
                    input_student, t_student, t, eps, data, real_data, condition=condition, neg_condition=neg_condition
                )

    def init_optimizers(self):
        """Initialize optimizers, lr_schedulers and grad_scalers"""
        super().init_optimizers()
        if not self.config.dmd_use_teacher_model_as_fake_score_estimate :
            # instantiate the optimizers for fake_score and discriminator
            self.fake_score_optimizer = instantiate(self.config.fake_score_optimizer, model=self.fake_score)
            # instantiate the lr schedulers for fake_score and discriminator
            self.fake_score_lr_scheduler = get_scheduler(self.fake_score_optimizer, self.config.fake_score_scheduler)

        if self.config.gan_loss_weight_gen > 0:
            # instantiate the discriminator in DMD2
            self.discriminator_optimizer = instantiate(self.config.discriminator_optimizer, model=self.discriminator)
            self.discriminator_lr_scheduler = get_scheduler(
                self.discriminator_optimizer, self.config.discriminator_scheduler
            )

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        """
        Get the optimizers for the current iteration
        Args:
            iteration (int): The current training iteration

        """
        if not self.config.dmd_use_teacher_model_as_fake_score_estimate:
            if iteration % self.config.student_update_freq == 0:
                return [self.net_optimizer]
            else:
                if self.config.gan_loss_weight_gen > 0:
                    return [self.fake_score_optimizer, self.discriminator_optimizer]
                else:
                    return [self.fake_score_optimizer]
        else:
            return [self.net_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler]:
        """
        Get the lr schedulers for the current iteration
        Args:
            iteration (int): The current training iteration

        """
        if not self.config.dmd_use_teacher_model_as_fake_score_estimate:
            if iteration % self.config.student_update_freq == 0:
                return [self.net_lr_scheduler]
            else:
                if self.config.gan_loss_weight_gen > 0:
                    return [self.fake_score_lr_scheduler, self.discriminator_lr_scheduler]
                else:
                    return [self.fake_score_lr_scheduler]
        else:
            return [self.net_lr_scheduler]
    @property
    def model_dict(self):
        """Return the model dict containing the student, fake_score, and discriminator models"""
        _model_dict = super().model_dict
        _model_dict["fake_score"] = self.fake_score
        if self.config.gan_loss_weight_gen > 0:
            _model_dict["discriminator"] = self.discriminator

        return _model_dict

    @property
    def optimizer_dict(self):
        """Return a dict containing all the optimizers"""
        _optimizer_dict = super().optimizer_dict
        _optimizer_dict["fake_score"] = self.fake_score_optimizer
        if self.config.gan_loss_weight_gen > 0:
            _optimizer_dict["discriminator"] = self.discriminator_optimizer

        return _optimizer_dict

    @property
    def scheduler_dict(self):
        """Return a dict containing all the lr schedulers"""
        _scheduler_dict = super().scheduler_dict
        _scheduler_dict["fake_score"] = self.fake_score_lr_scheduler
        if self.config.gan_loss_weight_gen > 0:
            _scheduler_dict["discriminator"] = self.discriminator_lr_scheduler

        return _scheduler_dict
