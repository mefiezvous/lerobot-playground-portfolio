# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for train.py entrypoint helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf


@pytest.mark.unit
def test_build_policy_act_uses_target() -> None:
    import train as train_module

    cfg = OmegaConf.create(
        {
            "_target_": "playground.policies.act_wrapper.ACTWrapper",
            "name": "act",
        }
    )
    fake_instance = MagicMock()
    fake_cls = MagicMock(return_value=fake_instance)
    with patch("train.get_class", return_value=fake_cls) as mock_get_class:
        policy = train_module._build_policy(cfg, device="cpu")
    mock_get_class.assert_called_once_with("playground.policies.act_wrapper.ACTWrapper")
    fake_cls.assert_called_once_with(cfg, device="cpu")
    assert policy is fake_instance


@pytest.mark.unit
def test_build_policy_diffusion_uses_target() -> None:
    import train as train_module

    cfg = OmegaConf.create(
        {
            "_target_": "playground.policies.diffusion_wrapper.DiffusionWrapper",
            "name": "diffusion",
        }
    )
    fake_instance = MagicMock()
    fake_cls = MagicMock(return_value=fake_instance)
    with patch("train.get_class", return_value=fake_cls) as mock_get_class:
        policy = train_module._build_policy(cfg, device="cpu")
    mock_get_class.assert_called_once_with("playground.policies.diffusion_wrapper.DiffusionWrapper")
    fake_cls.assert_called_once_with(cfg, device="cpu")
    assert policy is fake_instance


@pytest.mark.unit
def test_build_policy_missing_target_raises() -> None:
    import train as train_module

    cfg = OmegaConf.create({"name": "act"})
    with pytest.raises(ValueError, match="_target_"):
        train_module._build_policy(cfg, device="cpu")


@pytest.mark.unit
def test_build_policy_invalid_target_raises() -> None:
    import train as train_module

    cfg = OmegaConf.create({"_target_": "nonexistent.module.NotAClass", "name": "act"})
    with pytest.raises(ValueError, match="Failed to resolve policy '_target_'"):
        train_module._build_policy(cfg, device="cpu")


@pytest.mark.unit
def test_act_policy_config_has_target() -> None:
    """The shipped act.yaml must declare a _target_ for pluggable instantiation."""
    from pathlib import Path

    cfg = OmegaConf.load(Path(__file__).resolve().parent.parent / "configs" / "policy" / "act.yaml")
    assert cfg.policy._target_ == "playground.policies.act_wrapper.ACTWrapper"


@pytest.mark.unit
def test_diffusion_policy_config_has_target() -> None:
    """The shipped diffusion.yaml must declare a _target_ for pluggable instantiation."""
    from pathlib import Path

    cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent / "configs" / "policy" / "diffusion.yaml"
    )
    assert cfg.policy._target_ == "playground.policies.diffusion_wrapper.DiffusionWrapper"


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
