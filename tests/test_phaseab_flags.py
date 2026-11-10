# SPDX-License-Identifier: Apache-2.0
"""Phase B flags in DMD2Model (task2.md B0): guidance_high_t_cutoff and residual logging.

1. Flags off: losses and gradients equal the code before Phase B (dmd2.py at git HEAD) bit-exactly.
2. guidance_high_t_cutoff=0.0 equals guidance_scale=1.0; a cutoff above every t equals the flag off.
3. The cutoff is applied per sample.
4. Residual logging leaves the loss unchanged and writes one row per iteration over all accumulation calls.
"""

import gc
import importlib.util
import json
import os
import subprocess
import sys

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

from fastgen.methods import DMD2Model  # noqa: E402
from tests.test_dfd_flags import make_config, make_data, run_step  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAD_REV = os.environ.get("PHASEAB_BASE_REV", "HEAD")


def head_dmd2_class(tmp_dir):
    src = subprocess.run(["git", "show", f"{HEAD_REV}:fastgen/methods/distribution_matching/dmd2.py"],
                         cwd=REPO, check=True, capture_output=True, text=True).stdout
    assert "guidance_high_t_cutoff" not in src, "base revision already contains the Phase B code"
    path = os.path.join(tmp_dir, "dmd2_head.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("dmd2_head", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dmd2_head"] = mod
    spec.loader.exec_module(mod)
    return mod.DMD2Model


def build(cls, cfg, seed=0):
    gc.collect()
    torch.manual_seed(seed)
    m = cls(cfg)
    m.on_train_begin()
    m.init_optimizers()
    return m


def assert_same_step(m1, m2, it):
    l1, o1, g1 = run_step(m1, make_data(m1), it)
    l2, o2, g2 = run_step(m2, make_data(m2), it)
    assert l1.keys() == l2.keys()
    for k in l1:
        assert torch.equal(l1[k], l2[k]), (it, k)
    assert len(g1) == len(g2) and all(torch.equal(a, b) for a, b in zip(g1, g2)), it
    return l1


@pytest.mark.parametrize("iteration", [0, 1, 2, 3])  # student, fake-score, student, fake-score
@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_flags_off_match_head(tmp_path, iteration, p):
    Head = head_dmd2_class(str(tmp_path))
    new, old = build(DMD2Model, make_config(gen_real_replace_prob=p)), build(Head, make_config(gen_real_replace_prob=p))
    assert_same_step(new, old, iteration)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_cutoff_zero_equals_no_guidance(p):
    a = build(DMD2Model, make_config(gen_real_replace_prob=p, guidance_scale=3.0, guidance_high_t_cutoff=0.0))
    b = build(DMD2Model, make_config(gen_real_replace_prob=p, guidance_scale=1.0))
    for it in (0, 2, 4):
        assert_same_step(a, b, it)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_cutoff_above_all_t_equals_flag_off(p):
    a = build(DMD2Model, make_config(gen_real_replace_prob=p, guidance_high_t_cutoff=1.0e9))
    b = build(DMD2Model, make_config(gen_real_replace_prob=p))
    for it in (0, 2, 4):
        assert_same_step(a, b, it)


def test_cutoff_is_per_sample():
    m = build(DMD2Model, make_config(guidance_scale=3.0, guidance_high_t_cutoff=0.5))
    g = torch.Generator().manual_seed(5)
    x = torch.randn(3, 3, 8, 8, generator=g).to(m.device, m.precision)
    x0 = torch.randn(3, 3, 8, 8, generator=g).to(m.device, m.precision)
    neg = torch.zeros(3, 10).to(m.device, m.precision)
    t = torch.tensor([0.9, 0.2, 0.5], device=m.device, dtype=m.net.noise_scheduler.t_precision)
    out = m._apply_classifier_free_guidance(x, t, x0, neg_condition=neg)
    with torch.no_grad():
        x0_neg = m.teacher(x, t, condition=neg, fwd_pred_type="x0")
    assert torch.equal(out[0], x0[0]) and torch.equal(out[2], x0[2])  # t >= cutoff: w = 1
    assert torch.allclose(out[1], x0[1] + 2.0 * (x0[1] - x0_neg[1]))  # t < cutoff: w = guidance_scale


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_residual_logging_no_effect_and_rows(tmp_path, p):
    d = str(tmp_path / "res")
    mu = torch.randn(3, 8, 8)
    torch.save(mu, str(tmp_path / "mu_dir.pt"))
    on = build(DMD2Model, make_config(gen_real_replace_prob=p, dfd_residual_log_dir=d, dfd_residual_accum_rounds=2,
                                      dfd_residual_mu_dir_path=str(tmp_path / "mu_dir.pt")))
    off = build(DMD2Model, make_config(gen_real_replace_prob=p))
    for it in (2, 2, 3, 4, 4):  # two accumulation calls per student iteration, one fake-score step in between
        assert_same_step(on, off, it)
    rows = [json.loads(l) for l in open(os.path.join(d, "residuals.jsonl"))]
    assert [r["iteration"] for r in rows] == [2, 4] and all(r["complete"] for r in rows)
    samples = [json.loads(l) for l in open(os.path.join(d, "residual_samples_rank0.jsonl"))]
    assert len(samples) == 4 and all(s["replaced"] == (p == 1.0) for s in samples)
    for r in rows:
        for grouping in ("by_t_student", "by_t_score"):
            a = r[grouping]["all"]["all"]
            assert a["n"] == 2
            assert 0.0 <= a["common_mode_ratio"] <= 1.0 + 1e-6
            assert abs(a["cos_mu_dir"]) <= 1.0 + 1e-6
            assert a["diff_abs_mean"] > 0
            assert sum(v["all"]["n"] for k, v in r[grouping].items() if k != "all") == 2
            side = "replaced" if p == 1.0 else "not_replaced"
            assert r[grouping]["all"][side]["n"] == 2


def test_residual_single_sample_common_mode_is_one(tmp_path):
    d = str(tmp_path / "res")
    m = build(DMD2Model, make_config(dfd_residual_log_dir=d, dfd_residual_accum_rounds=1))
    run_step(m, make_data(m), 2)
    row = json.loads(open(os.path.join(d, "residuals.jsonl")).readline())
    a = row["by_t_student"]["all"]["all"]
    assert a["n"] == 1 and abs(a["common_mode_ratio"] - 1.0) < 1e-5 and "proj_mu_dir" not in a
