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
    with patch("train.ACTWrapper") as mock_act:
        mock_act.return_value = MagicMock()
        policy = train_module._build_policy(cfg, device="cpu")
    mock_act.assert_called_once_with(cfg, device="cpu")
    assert policy is mock_act.return_value


@pytest.mark.unit
def test_build_policy_diffusion() -> None:
    import train as train_module

    cfg = OmegaConf.create({"name": "diffusion"})
    with patch("train.DiffusionWrapper") as mock_diff:
        mock_diff.return_value = MagicMock()
        policy = train_module._build_policy(cfg, device="cpu")
    mock_diff.assert_called_once_with(cfg, device="cpu")
    assert policy is mock_diff.return_value


@pytest.mark.unit
def test_build_policy_unknown_raises() -> None:
    import train as train_module

    cfg = OmegaConf.create({"name": "unknown_policy"})
    with pytest.raises(ValueError, match="Unknown policy name"):
        train_module._build_policy(cfg, device="cpu")
