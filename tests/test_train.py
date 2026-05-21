# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for train.py entrypoint helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf


@pytest.mark.unit
def test_build_policy_act() -> None:
    import train as train_module

    cfg = OmegaConf.create({"name": "act"})
    with patch("train.ACTWrapper") as MockACT:
        MockACT.return_value = MagicMock()
        policy = train_module._build_policy(cfg, device="cpu")
    MockACT.assert_called_once_with(cfg, device="cpu")
    assert policy is MockACT.return_value


@pytest.mark.unit
def test_build_policy_diffusion() -> None:
    import train as train_module

    cfg = OmegaConf.create({"name": "diffusion"})
    with patch("train.DiffusionWrapper") as MockDiff:
        MockDiff.return_value = MagicMock()
        policy = train_module._build_policy(cfg, device="cpu")
    MockDiff.assert_called_once_with(cfg, device="cpu")
    assert policy is MockDiff.return_value


@pytest.mark.unit
def test_build_policy_unknown_raises() -> None:
    import train as train_module

    cfg = OmegaConf.create({"name": "unknown_policy"})
    with pytest.raises(ValueError, match="Unknown policy name"):
        train_module._build_policy(cfg, device="cpu")
