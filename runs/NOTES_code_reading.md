# Code-reading notes (DFD / FastGen, Wan2.1-1.3B T2V)

Repo: `csy2077/data-forcing-distillation` @ `de6416c` (origin `git@github.com:csy2077/data-forcing-distillation.git`).
Files read: `fastgen/methods/distribution_matching/dmd2.py`, `fastgen/methods/common_loss.py`,
`fastgen/methods/model.py`, `fastgen/networks/noise_schedule.py` (RF), `fastgen/trainer.py`,
`fastgen/utils/checkpointer.py`, `fastgen/configs/methods/config_dmd2.py`,
`fastgen/configs/experiments/WanT2V/config_dmd2.py`, `fastgen/configs/net.py`.

---

## 1. `forward_kl_ensemble=True` in `variational_score_distillation_loss` (spec §1.1 task)

`common_loss.py:63-106`:

```python
diff_abs_mean = (gen_data_fp32 - teacher_x0_fp32).abs().mean(dim=dims, keepdim=True)
w = 1 / (diff_abs_mean + 1e-6)
if forward_kl_ensemble == False:
    vsd_grad = (fake_score_x0 - teacher_x0) * w
else:
    vsd_grad = (fake_score_x0 - teacher_x0) * w      # identical to the other branch
pseudo_target = gen_data - vsd_grad
loss = 0.5 * F.mse_loss(gen_data, pseudo_target)
```

**Conclusion: `forward_kl_ensemble` has no effect.** Both branches are byte-identical, so the flag
changes nothing in the loss or the gradient. The `## TODO: here do the minus signal change` comment at
`dmd2.py:235` suggests the authors planned a sign flip for the replaced case, but it was never implemented.
The released DFD result is therefore produced by the plain DMD VSD expression, with only the
*inputs* changed:

- `teacher_x0` = CFG teacher x0-prediction evaluated at the **noised real point**
  `x_t^real = (1-t)·x_real + t·eps` (same `eps`, same `t` as the gen point).
- `fake_score_x0` = fake score evaluated at the **noised gen point** `x_t^gen = (1-t)·G + t·eps`
  (because `model.forward_kl=False` by default; `forward_kl=True` would move the fake score to the
  real point too).
- The per-sample normalizer `w = 1 / mean|G − teacher_x0|` is also computed with the teacher prediction
  **at the real point**, so when replacement happens `w` measures the distance between the generation and
  the teacher's denoised *real* sample, not the teacher's denoised *generation*.
- Gradient w.r.t. `gen_data`: `∂loss/∂G = vsd_grad / N` (N = number of elements), i.e. the student is
  pushed along `−(fake_x0(x_t^gen) − teacher_cfg_x0(x_t^real))·w`.
  Since `x_t^real − x_t^gen = (1−t)(x_real − G)`, the "teacher score discrepancy" term that DFD adds is
  `teacher_x0(x_t^real) − teacher_x0(x_t^gen)`; its size is controlled by `(1−t)·‖x_real − G‖` and the
  teacher's Lipschitz constant (spec A.2 conditions 1–2).

## 2. The replacement branch itself (`dmd2.py:233-251`)

```python
if self.config.gen_real_replace_prob > 0:
    if (torch.rand(()) < p) and (post_train == False):        # DMD-stage variant: 1 sample replaced
        ...
    elif (torch.rand(()) < p) and (post_train == True):       # post-train variant: whole batch replaced
        gen_idx = torch.randint(...).item()                   # drawn but unused
        real_idx = torch.randint(...).item()                  # drawn but unused
        gen_data_teacher_score = gen_data + (real_data - gen_data).detach()
        forward_kl_ensemble = True                            # no-op, see §1
```

Observations relevant to reproducibility and to "defaults must be bit-exact":

- With `post_train=True`, **two** `torch.rand` draws happen per student micro-step: the first `if` always
  evaluates `torch.rand(()) < p` before short-circuiting on `post_train == False`. The effective replace
  probability is still `p = 0.5` (independent second draw), but any new code that changes the number or order
  of RNG calls breaks bit-exactness against the original run. The two `randint` calls in the post-train
  branch are also consumed only when replacement happens.
- Pairing: `real_data[i]` is paired with `gen_data[i]` (same caption `c_i`); no shuffling.
- Granularity: the decision is per **rank and per gradient-accumulation micro-batch** (the trainer seeds with
  `by_rank=True`, so every GPU draws independently). With 8 GPUs × per-GPU batch 2 × accum 1, a student step
  replaces a random subset of the 8 micro-batches (each all-or-nothing, 2 samples), not the whole global batch.
- The straight-through `gen + (real − gen).detach()` has value `real_data`. Its gradient path to `G` is
  irrelevant for the VSD loss (everything built from it is used under `no_grad` / `.detach()`), but see §4 for
  memory.

## 3. Multi-step student: what `gen_data_from_net` does (spec §1.1 task)

`_generate_noise_and_time` (`dmd2.py:79-122`), with `student_sample_steps=4`,
`t_list=[0.999, 0.937, 0.833, 0.624, 0.0]`:

```python
t_student = sample_from_t_list(...)          # uniform over t_list[:-1] = {0.999, 0.937, 0.833, 0.624}
input_student = forward_process(real_data, eps_student, t_student)   # (1-t)·x_real + t·eps_student
```

`gen_data_from_net` (`model.py:306-313`) is **one** forward pass:
`gen_data = self.net(input_student, t_student, condition, fwd_pred_type="x0")`.
It does **not** run the remaining steps of the 4-step schedule (no backward simulation / self-rollout).
Training therefore teaches the student "one-step x0 prediction from a noised *dataset* sample at one of the
4 schedule times", while at inference the inputs to steps 2–4 are noised versions of the student's *own*
predictions (`_student_sample_loop`, `student_sample_type="ode"`).

Consequences to discuss in every experiment's NOTES:

- **Switching the dataset changes the student's input distribution**, not only the teacher-score target.
  With real-world data, the student's training inputs become noised real-world videos.
- **The (G, x_real) pair is already coupled.** `G_i` is predicted from a noised copy of `x_real,i`. At
  `t_student=0.624` the input still carries 37.6 % of the real signal, so `‖x_real − G‖` (spec condition 1's δ)
  is small by construction for 3 of the 4 schedule times, and near-random only at `t_student=0.999`.
  This matters for E4 (inversion) and E4b (OT): the pairing is not a random pairing to begin with, and
  replacing `eps_student` by `z_inv` also changes `input_student` for `t_student < 0.999`.
- `t_student` and `t` (the DMD noise level) are sampled independently; `eps_student` and `eps` are independent.

## 4. Teacher calls, CFG, memory (`dmd2.py:124-205`, `259-297`)

- `gan_loss_weight_gen=0.03 > 0`, so `_compute_teacher_prediction_gan_loss` runs the teacher **twice on every
  student step**, whether or not replacement happened:
  1. `teacher(x_t^real-or-gen, feature_indices=...)` → returned as `teacher_x0` for the VSD loss.
  2. `teacher(x_t^gen, feature_indices=...)` → only its features `fake_feat` are used (GAN generator loss);
     its x0 output is discarded.
  Without replacement both inputs are identical (redundant compute). So "DFD replacement costs an extra teacher
  forward" is not true here: the cost is the same with or without replacement.
- Neither call is under `no_grad`. Call 2 needs the graph (GAN loss backprops through teacher features into
  `G`). Call 1's input also carries a graph back to `G` (straight-through), so its activations are kept until
  the non-detached output is freed on return. The `.detach()` on return prevents any gradient from flowing
  through it. Peak memory is therefore similar with and without replacement. The frame probe (§2.5) should
  still include both kinds of step.
- CFG (`_apply_classifier_free_guidance`, under `no_grad`): the unconditional branch is evaluated at
  `perturbed_data_teacher_score` (the real point when replaced), with `neg_condition`;
  `x0 = x0_cond + (w−1)(x0_cond − x0_uncond)`, `w = 5.0`. Confirms spec §1.1.
- Fake score: `fake_score(x_t^gen)` under `no_grad` (`forward_kl=False`).
- Other flags that exist but are off by default: `dmd_continue_remove_fake_score_discriminator`,
  `dmd_use_teacher_model_as_fake_score_estimate`, `forward_kl` (all `False`).

## 5. Update schedule (`trainer.py:190`, `dmd2.py:67-77, 511-520`)

- `student_update_freq = 5` (method default; not overridden in the WanT2V config).
- Training loop: `for iter_cur in range(iter_start + 1, max_iter)`; student update iff `iter_cur % 5 == 0`,
  otherwise one fake_score + discriminator update.
- **The official 40-step DFD result (`ours_wan_t2v_vipe/0000040`) therefore has only 8 generator updates**
  (iters 5, 10, …, 40) and 32 critic updates. This is why seed and critic init matter so much (spec §5, risk 2).
- `resume_iter` from the pretrained `.pth` (1000 for `dmd2_t2v_vipe`) is set on the model but only used by the
  EMA and consistency-time schedule callbacks. EMA is off for this config (`use_ema=False`). Post-train
  iterations count from 0, which matches the released `0000040` naming.

## 6. Model construction (`dmd2.py:40-65`, `model.py:176-255`)

- Teacher: `instantiate(Wan_1_3B_Config)` → loads `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` from HF
  (`load_pretrained=True`). `pretrained_model_path=""` for Wan, so no extra teacher checkpoint is loaded.
- fake_score: `instantiate(teacher_config)`, i.e. the same HF weights = **teacher copy**.
  (The explicit `load_state_dict(teacher.state_dict())` is skipped because `pretrained_model_path` is empty,
  but the result is the same since both come from the same HF checkpoint.)
- discriminator: `instantiate(Discriminator_Wan_1_3B_Config)`, `disc_type="multiscale_down_mlp_large"`,
  `feature_indices=[15, 22, 29]`, **random init**.
- Student `net`: same HF weights, then overwritten by `pretrained_ckpt_path` via `load_pretrained_ckpt`.
- Gradient checkpointing: `Wan(disable_grad_ckpt=False)` default, and `Wan_1_3B_Config` does not override it
  → **on** (relevant for §2.5 step 2).

## 7. Checkpoint loading (`trainer.py:243-271`, `checkpointer.py:340-460`)

- `load_pretrained_ckpt` loops over `pretrained_ckpt_key_map` (`{model attr: ckpt key}`). For each entry whose
  model attribute exists, it calls `FSDPCheckpointer.load({k_ckpt: module}, path=pretrained_ckpt_path)`.
- Expected on-disk layout for prefix `<dir>/0001000`: `0001000.pth` (scheduler / grad_scaler / callbacks /
  iteration) and one DCP **directory** per key, `0001000.<key>_model/` (`.metadata` + `__*_0.distcp`).
  The released `.net_model.zip` must be **unzipped** (the archive already contains `0001000.net_model/`).
- **A missing key raises; it is not skipped:**
  `assert os.path.exists(f"{path}.{k}_model"), f"Key {k} does not exist in FSDP model dict"`
  (`checkpointer.py:388`). With the README key map `{net: net, fake_score: fake_score, discriminator:
  discriminator}` and a net-only checkpoint, `net` loads and then `fake_score` fails with `AssertionError`.
  The only skip path is when the *model* lacks the attribute (warning "Model does not have submodule").
- Within one key, `dcp.load` uses the default planner (`allow_partial_load=False`): a model key absent from the
  checkpoint **raises**, but extra keys in the checkpoint are **silently ignored**. So "unexpected keys" must
  be checked separately (see `runs/phase0/inspect_ckpt_keys.py`).

## 8. Noise schedule / score conversion (for F2)

`RFNoiseSchedule`: `x_t = (1−t)·x0 + t·eps`, so `alpha_t = 1−t`, `sigma_t = t`.
From an x0-prediction x̂0: `epŝ = (x_t − (1−t)·x̂0)/t`, score
`s(x_t) = −epŝ/sigma_t = ((1−t)·x̂0 − x_t)/t²`, which matches the spec formula `s = (α_t·x̂0 − x_t)/σ_t²`.
For two points sharing (eps, t): `s(x_t^real) − s(x_t^gen) = [(1−t)(x̂0^real − x̂0^gen) − (1−t)(x_real − G)]/t²`.
