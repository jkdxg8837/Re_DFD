# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `trainer.checkpointer.full_save_iter` (DFD experiments, default-off).

1. Default (0): `Trainer.save_checkpoint` passes exactly `model.model_dict` / `model.optimizer_dict` to the
   checkpointer at every iteration, i.e. the original behavior.
2. full_save_iter=20, max_iter=40: iterations 20 and 40 save everything; 5, 10, 15, 25, 30, 35 save only
   {net, EMA} and no optimizer state.

Usage: pytest tests/test_full_save_iter.py -v
"""

from types import SimpleNamespace
from unittest import mock

import torch

from fastgen.configs.config import BaseCheckpointerConfig
from fastgen.trainer import Trainer


class _Model:
    def __init__(self):
        self.net = torch.nn.Linear(2, 2)
        self.fake_score = torch.nn.Linear(2, 2)
        self.ema_dict = {}
        self.model_dict = torch.nn.ModuleDict({"net": self.net, "fake_score": self.fake_score})
        self.optimizer_dict = {"net": object(), "fake_score": object()}
        self.scheduler_dict = {"net": object()}
        self.grad_scaler = None


def _trainer(full_save_iter=None, max_iter=40):
    ckpt_cfg = BaseCheckpointerConfig()
    if full_save_iter is not None:
        ckpt_cfg.full_save_iter = full_save_iter
    return SimpleNamespace(
        config=SimpleNamespace(trainer=SimpleNamespace(checkpointer=ckpt_cfg, max_iter=max_iter)),
        callbacks=mock.MagicMock(),
        checkpointer=mock.MagicMock(save=mock.MagicMock(return_value="p")),
    )


def _saved(trainer, model, iteration):
    Trainer.save_checkpoint(trainer, model, iteration)
    args, kwargs = trainer.checkpointer.save.call_args
    return args[0], kwargs["optimizer_dict"], kwargs["scheduler_dict"]


def test_default_is_original_behavior():
    assert BaseCheckpointerConfig().full_save_iter == 0
    model = _Model()
    for it in (5, 10, 15, 20, 40):
        md, od, sd = _saved(_trainer(), model, it)
        assert md is model.model_dict and od is model.optimizer_dict and sd is model.scheduler_dict


def test_net_only_intermediate_saves():
    model = _Model()
    for it in (5, 10, 15, 25, 30, 35):
        md, od, sd = _saved(_trainer(full_save_iter=20), model, it)
        assert list(md.keys()) == ["net"] and md["net"] is model.net
        assert od is None and sd is model.scheduler_dict
    for it in (20, 40):
        md, od, _ = _saved(_trainer(full_save_iter=20), model, it)
        assert md is model.model_dict and od is model.optimizer_dict


def test_final_save_is_always_full():
    model = _Model()
    md, od, _ = _saved(_trainer(full_save_iter=20, max_iter=45), model, 45)
    assert md is model.model_dict and od is model.optimizer_dict
