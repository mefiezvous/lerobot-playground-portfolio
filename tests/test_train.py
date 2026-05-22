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


@pytest.mark.unit
def test_build_dataloader_act_delta_timestamps() -> None:
    import train as train_module

    dataset_cfg = OmegaConf.create({"repo_id": "test/dataset", "root": None})
    policy_cfg = OmegaConf.create({"n_obs_steps": 1, "chunk_size": 100})

    with (
        patch("train._LEROBOT_AVAILABLE", True),
        patch("train.LeRobotDataset") as mock_ds,
        patch("train.DataLoader") as mock_dl,
    ):
        mock_ds.return_value = MagicMock()
        mock_dl.return_value = MagicMock()
        train_module._build_dataloader(dataset_cfg, policy_cfg, batch_size=64, fps=20)

    call_kwargs = mock_ds.call_args.kwargs
    dt = call_kwargs["delta_timestamps"]
    assert dt["observation.state"] == [0.0]
    assert len(dt["action"]) == 100
    assert dt["action"][0] == pytest.approx(0.0)
    assert dt["action"][1] == pytest.approx(0.05)


@pytest.mark.unit
def test_build_dataloader_diffusion_delta_timestamps() -> None:
    import train as train_module

    dataset_cfg = OmegaConf.create({"repo_id": "test/dataset", "root": None})
    policy_cfg = OmegaConf.create({"n_obs_steps": 2, "n_action_steps": 8})

    with (
        patch("train._LEROBOT_AVAILABLE", True),
        patch("train.LeRobotDataset") as mock_ds,
        patch("train.DataLoader") as mock_dl,
    ):
        mock_ds.return_value = MagicMock()
        mock_dl.return_value = MagicMock()
        train_module._build_dataloader(dataset_cfg, policy_cfg, batch_size=64, fps=20)

    call_kwargs = mock_ds.call_args.kwargs
    dt = call_kwargs["delta_timestamps"]
    assert dt["observation.state"] == pytest.approx([-0.05, 0.0])
    assert len(dt["action"]) == 8
