# SPDX-License-Identifier: Apache-2.0
"""DFD experiment flags in DMD2Model (dfd_student_input_key, dfd_diagnostics_every).

1. Flags off: losses and gradients equal the original DFD dmd2.py (loaded from the base commit) bit-exactly.
2. dfd_student_input_key: the student input is built from the second stream, not from data["real"].
3. dfd_diagnostics_every: the loss is unchanged (diagnostics are no-grad and use their own RNG) and rows are written.
"""

import gc
import importlib.util
import json
import os
import subprocess
import sys

import pytest

# bit-exact comparisons need deterministic GPU kernels (conv/matmul backward otherwise uses atomics)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

from fastgen.configs.config_utils import override_config_with_opts
from fastgen.configs.methods.config_dmd2 import ModelConfig
from fastgen.methods import DMD2Model

BASE_COMMIT = "de6416c"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def original_dmd2_class(tmp_dir):
    src = subprocess.run(["git", "show", f"{BASE_COMMIT}:fastgen/methods/distribution_matching/dmd2.py"],
                         cwd=REPO, check=True, capture_output=True, text=True).stdout
    path = os.path.join(tmp_dir, "dmd2_original.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("dmd2_original", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dmd2_original"] = mod
    spec.loader.exec_module(mod)
    return mod.DMD2Model


def make_config(**over):
    c = ModelConfig()
    c.net = override_config_with_opts(c.net, ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1"])
    c.discriminator = override_config_with_opts(c.discriminator, ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"])
    c.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c.precision = "float32"
    c.pretrained_model_path = ""
    c.student_update_freq = 2
    c.input_shape = [3, 8, 8]
    c.gan_loss_weight_gen = 0.0
    c.post_train = True
    c.gen_real_replace_prob = 0.5
    c.guidance_scale = 3.0
    c.student_sample_steps = 2
    for k, v in over.items():
        setattr(c, k, v)
    return c


def build(cls, cfg, seed=0):
    gc.collect()
    torch.manual_seed(seed)
    m = cls(cfg)
    m.on_train_begin()
    m.init_optimizers()
    return m


def make_data(model, seed=123, extra=None):
    g = torch.Generator().manual_seed(seed)
    labels = torch.nn.functional.one_hot(torch.randint(0, 10, (1,), generator=g), num_classes=10)
    d = {
        "real": torch.randn(1, 3, 8, 8, generator=g).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": torch.zeros(1, 10).to(model.device, model.precision),
    }
    d.update(extra or {})
    return d


def run_step(model, data, iteration, seed=7):
    torch.manual_seed(seed)
    loss_map, outputs = model.single_train_step(data, iteration)
    loss_map["total_loss"].backward()
    grads = [p.grad.detach().clone() for p in model.net.parameters() if p.grad is not None]
    return loss_map, outputs, grads


@pytest.mark.parametrize("iteration", [0, 1, 2, 3])  # student, fake-score, student, fake-score
@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_flags_off_match_original(tmp_path, iteration, p):
    Orig = original_dmd2_class(str(tmp_path))
    cfg = make_config(gen_real_replace_prob=p)
    new, old = build(DMD2Model, cfg), build(Orig, make_config(gen_real_replace_prob=p))
    for a, b in zip(new.net.parameters(), old.net.parameters()):
        assert torch.equal(a, b)
    l1, o1, g1 = run_step(new, make_data(new), iteration)
    l2, o2, g2 = run_step(old, make_data(old), iteration)
    assert l1.keys() == l2.keys()
    for k in l1:
        assert torch.equal(l1[k], l2[k]), k
    for k in o1:
        if isinstance(o1[k], torch.Tensor):
            assert torch.equal(o1[k], o2[k]), k
    assert len(g1) == len(g2) and all(torch.equal(a, b) for a, b in zip(g1, g2))


def capture_student_input(model):
    seen = []
    orig = model.gen_data_from_net

    def wrapped(input_student, t_student, **kw):
        seen.append((input_student.detach().clone(), t_student.detach().clone()))
        return orig(input_student, t_student, **kw)

    model.gen_data_from_net = wrapped
    return seen


@pytest.mark.parametrize("iteration", [0, 1])  # student update and fake-score update both use the student input
def test_student_input_key_uses_second_stream(iteration):
    c = 1.0e6
    m_on = build(DMD2Model, make_config(dfd_student_input_key="student_real"))
    m_off = build(DMD2Model, make_config())
    d = make_data(m_on)
    d["real"] = torch.zeros_like(d["real"])
    d["student_real"] = torch.full_like(d["real"], c)
    seen_on, seen_off = capture_student_input(m_on), capture_student_input(m_off)
    torch.manual_seed(7)
    m_on.single_train_step(dict(d), iteration)
    torch.manual_seed(7)
    m_off.single_train_step(dict(d), iteration)
    (x_on, t_on), (x_off, t_off) = seen_on[0], seen_off[0]
    ns = m_on.net.noise_scheduler
    # same t_student; input = alpha(t) * c + sigma(t) * eps' with c from data["student_real"], vs sigma * eps from real=0
    assert torch.equal(t_on, t_off)
    alpha = ns.alpha(t_on).float().item()
    assert abs(x_on.float().mean().item() / (alpha * c) - 1) < 1e-2
    assert x_off.float().abs().mean().item() < 1e-3 * alpha * c


def test_student_input_key_single_step_is_noop():
    m = build(DMD2Model, make_config(student_sample_steps=1, dfd_student_input_key="student_real"))
    d = make_data(m)
    d["student_real"] = torch.full_like(d["real"], 1.0e4)
    ref = build(DMD2Model, make_config(student_sample_steps=1))
    l1, _, _ = run_step(m, d, 0)
    l2, _, _ = run_step(ref, make_data(ref), 0)
    assert torch.equal(l1["total_loss"], l2["total_loss"])


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_diagnostics_do_not_change_loss(tmp_path, p):
    diag = str(tmp_path / "diag")
    m_on = build(DMD2Model, make_config(gen_real_replace_prob=p, dfd_diagnostics_every=1, dfd_diagnostics_dir=diag))
    m_off = build(DMD2Model, make_config(gen_real_replace_prob=p))
    for it in (2, 3, 4):  # student, fake-score, student
        l1, _, g1 = run_step(m_on, make_data(m_on), it)
        l2, _, g2 = run_step(m_off, make_data(m_off), it)
        for k in l1:
            assert torch.equal(l1[k], l2[k]), (it, k)
        assert all(torch.equal(a, b) for a, b in zip(g1, g2))
    rows = [json.loads(l) for l in open(os.path.join(diag, "rank0.jsonl"))]
    assert [r["iteration"] for r in rows] == [2, 4]
    assert all(r["replaced"] == (p == 1.0) for r in rows)
    for r in rows:
        for k in ("delta_sq", "score_diff_sq", "xt_diff_norm", "lipschitz_hat", "teacher_x0_diff_sq"):
            assert r[k] == r[k] and r[k] >= 0, (k, r[k])  # finite, non-negative
    sk = [f for f in os.listdir(diag) if f.startswith("sketch_")]
    assert len(sk) == 2
    s = torch.load(os.path.join(diag, sorted(sk)[0]))
    assert s["sketch"].shape == (1, m_on.config.dfd_diagnostics_sketch_dim)
    # count sketch preserves the squared norm in expectation; with D >> N it is exact up to collisions
    assert abs(float(s["sketch"].pow(2).sum()) / rows[0]["score_diff_sq"] - 1) < 0.2


def test_diagnostics_every_skips_steps(tmp_path):
    diag = str(tmp_path / "diag")
    m = build(DMD2Model, make_config(dfd_diagnostics_every=2, dfd_diagnostics_dir=diag))
    for it in (2, 4, 6, 8):  # student updates 1..4 -> logged at 1 and 3
        run_step(m, make_data(m), it)
    rows = [json.loads(l) for l in open(os.path.join(diag, "rank0.jsonl"))]
    assert [r["student_step"] for r in rows] == [1, 3]


# ---------------------------------------------------------------------------------------------------------------
# E3: dfd_cfg_uncond_on_gen
# ---------------------------------------------------------------------------------------------------------------
def capture_cfg_inputs(model):
    seen = []
    orig = model._apply_classifier_free_guidance

    def wrapped(perturbed, t, teacher_x0, neg_condition=None):
        seen.append(perturbed.detach().clone())
        return orig(perturbed, t, teacher_x0, neg_condition=neg_condition)

    model._apply_classifier_free_guidance = wrapped
    return seen


def test_e3_no_replacement_matches_off():
    # p = 0: never replaced, so the flag must not change anything
    m_on = build(DMD2Model, make_config(gen_real_replace_prob=0.0, dfd_cfg_uncond_on_gen=True))
    m_off = build(DMD2Model, make_config(gen_real_replace_prob=0.0))
    l1, _, g1 = run_step(m_on, make_data(m_on), 0)
    l2, _, g2 = run_step(m_off, make_data(m_off), 0)
    assert torch.equal(l1["total_loss"], l2["total_loss"])
    assert all(torch.equal(a, b) for a, b in zip(g1, g2))


def test_e3_uncond_at_gen_point_when_replaced():
    m_on = build(DMD2Model, make_config(gen_real_replace_prob=1.0, dfd_cfg_uncond_on_gen=True))
    m_off = build(DMD2Model, make_config(gen_real_replace_prob=1.0))
    s_on, s_off = capture_cfg_inputs(m_on), capture_cfg_inputs(m_off)
    captured = {}
    for m, name in ((m_on, "on"), (m_off, "off")):
        orig_fp = m.net.noise_scheduler.forward_process
        calls = []

        def fp(x, e, t, _orig=orig_fp, _calls=calls):
            out = _orig(x, e, t)
            _calls.append(out.detach().clone())
            return out

        m.net.noise_scheduler.forward_process = fp
        captured[name] = calls
    l_on, _, _ = run_step(m_on, make_data(m_on), 0)
    l_off, _, _ = run_step(m_off, make_data(m_off), 0)
    # forward_process calls in the student step: [input_student, perturbed_data (gen), perturbed_data_teacher_score (real)]
    gen_pt_on, real_pt_on = captured["on"][-2], captured["on"][-1]
    real_pt_off = captured["off"][-1]
    assert torch.equal(s_on[0], gen_pt_on) and not torch.equal(s_on[0], real_pt_on)
    assert torch.equal(s_off[0], real_pt_off)
    assert not torch.equal(l_on["total_loss"], l_off["total_loss"])


def test_e3_with_diagnostics_loss_unchanged(tmp_path):
    diag = str(tmp_path / "diag")
    m_d = build(DMD2Model, make_config(gen_real_replace_prob=1.0, dfd_cfg_uncond_on_gen=True, dfd_diagnostics_every=1,
                                       dfd_diagnostics_dir=diag))
    m = build(DMD2Model, make_config(gen_real_replace_prob=1.0, dfd_cfg_uncond_on_gen=True))
    l1, _, g1 = run_step(m_d, make_data(m_d), 2)
    l2, _, g2 = run_step(m, make_data(m), 2)
    assert torch.equal(l1["total_loss"], l2["total_loss"])
    assert all(torch.equal(a, b) for a, b in zip(g1, g2))
    rows = [json.loads(l) for l in open(os.path.join(diag, "rank0.jsonl"))]
    assert len(rows) == 1 and rows[0]["replaced"]


# ---------------------------------------------------------------------------------------------------------------
# E4: dfd_inverted_noise_key
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("iteration", [0, 1])
def test_e4_inverted_noise_multistep(iteration):
    m = build(DMD2Model, make_config(dfd_inverted_noise_key="z_inv"))
    d = make_data(m)
    g = torch.Generator().manual_seed(5)
    d["z_inv"] = torch.randn(d["real"].shape, generator=g).to(m.device, m.precision)
    seen = capture_student_input(m)
    torch.manual_seed(7)
    m.single_train_step(dict(d), iteration)
    x, t = seen[0]
    expect = m.net.noise_scheduler.forward_process(d["real"], d["z_inv"], t)
    assert torch.equal(x, expect)


def test_e4_inverted_noise_single_step():
    m = build(DMD2Model, make_config(student_sample_steps=1, dfd_inverted_noise_key="z_inv"))
    d = make_data(m)
    d["z_inv"] = torch.randn_like(d["real"])
    seen = capture_student_input(m)
    torch.manual_seed(7)
    m.single_train_step(dict(d), 0)
    assert torch.equal(seen[0][0], m.net.noise_scheduler.latents(noise=d["z_inv"]))


def test_e4_rejects_e2s_arm():
    m = build(DMD2Model, make_config(dfd_inverted_noise_key="z_inv", dfd_student_input_key="student_real"))
    d = make_data(m)
    d["z_inv"] = torch.randn_like(d["real"])
    d["student_real"] = torch.randn_like(d["real"])
    with pytest.raises(AssertionError):
        m.single_train_step(d, 0)
