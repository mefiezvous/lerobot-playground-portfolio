# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for ACTWrapper and DiffusionWrapper.

All tests are pure unit tests: lerobot ACTPolicy / DiffusionPolicy are mocked
so no GPU or real lerobot weights are required.

Mock strategy
-------------
Each test class uses a module-level fixture that patches the lerobot class *in
the wrapper's own namespace* plus ``_LEROBOT_AVAILABLE`` so the constructors
proceed even when lerobot is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def act_cfg() -> Any:
    return OmegaConf.create(
        {
            "name": "act",
            "pretrained_repo": None,
            "n_obs_steps": 1,
            "n_action_steps": 100,
            "chunk_size": 100,
            "input_shapes": {"observation.state": [3]},
            "output_shapes": {"action": [8]},
        }
    )


@pytest.fixture
def act_cfg_pretrained(act_cfg: Any) -> Any:
    cfg = OmegaConf.to_container(act_cfg, resolve=True)
    assert isinstance(cfg, dict)
    cfg["pretrained_repo"] = "mefiezvous/cube-reach-v1-act"
    return OmegaConf.create(cfg)


@pytest.fixture
def diffusion_cfg() -> Any:
    return OmegaConf.create(
        {
            "name": "diffusion",
            "pretrained_repo": None,
            "n_obs_steps": 2,
            "n_action_steps": 8,
            "num_inference_steps": 10,
            "input_shapes": {"observation.state": [3]},
            "output_shapes": {"action": [8]},
        }
    )


@pytest.fixture
def diffusion_cfg_pretrained(diffusion_cfg: Any) -> Any:
    cfg = OmegaConf.to_container(diffusion_cfg, resolve=True)
    assert isinstance(cfg, dict)
    cfg["pretrained_repo"] = "mefiezvous/cube-reach-v1-diffusion"
    return OmegaConf.create(cfg)


def _make_mock_lerobot_policy() -> MagicMock:
    """Return a MagicMock that behaves like a lerobot policy module."""
    mock = MagicMock()
    mock.state_dict.return_value = {"layer": torch.zeros(4)}
    # select_action returns (batch=1, action_dim=8) tensor
    mock.select_action.return_value = torch.zeros(1, 8)
    # forward returns loss dict
    mock.forward.return_value = {"loss": torch.tensor(0.42)}
    return mock


# ---------------------------------------------------------------------------
# ACTWrapper tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestACTWrapper:
    """Unit tests for ACTWrapper — lerobot ACTPolicy is fully mocked."""

    def _make_wrapper(self, cfg: Any, mock_policy_instance: MagicMock) -> Any:
        """Helper: build an ACTWrapper with all lerobot deps patched."""
        from playground.policies.act_wrapper import ACTWrapper  # noqa: PLC0415

        with (
            patch("playground.policies.act_wrapper.ACTPolicy") as mock_cls,
            patch("playground.policies.act_wrapper.ACTConfig"),
            patch("playground.policies.act_wrapper._LEROBOT_AVAILABLE", True),
        ):
            mock_cls.return_value = mock_policy_instance
            wrapper = ACTWrapper(cfg, device="cpu")
        return wrapper

    def test_fresh_init_builds_policy_from_config(self, act_cfg: Any) -> None:
        from playground.policies.act_wrapper import ACTWrapper  # noqa: PLC0415

        mock_policy = _make_mock_lerobot_policy()
        with (
            patch("playground.policies.act_wrapper.ACTPolicy") as mock_cls,
            patch("playground.policies.act_wrapper.ACTConfig") as mock_cfg_cls,
            patch("playground.policies.act_wrapper._LEROBOT_AVAILABLE", True),
        ):
            mock_cls.return_value = mock_policy
            ACTWrapper(act_cfg, device="cpu")
            mock_cfg_cls.assert_called_once()
            mock_cls.assert_called_once()

    def test_pretrained_init_calls_from_pretrained(self, act_cfg_pretrained: Any) -> None:
        from playground.policies.act_wrapper import ACTWrapper  # noqa: PLC0415

        mock_policy = _make_mock_lerobot_policy()
        with (
            patch("playground.policies.act_wrapper.ACTPolicy") as mock_cls,
            patch("playground.policies.act_wrapper._LEROBOT_AVAILABLE", True),
        ):
            mock_cls.from_pretrained.return_value = mock_policy
            ACTWrapper(act_cfg_pretrained, device="cpu")
            mock_cls.from_pretrained.assert_called_once_with("mefiezvous/cube-reach-v1-act")

    def test_select_action_returns_numpy_array(self, act_cfg: Any) -> None:
        from playground.policies.act_wrapper import ACTWrapper  # noqa: PLC0415

        mock_policy = _make_mock_lerobot_policy()
        with (
            patch("playground.policies.act_wrapper.ACTPolicy") as mock_cls,
            patch("playground.policies.act_wrapper.ACTConfig"),
            patch("playground.policies.act_wrapper._LEROBOT_AVAILABLE", True),
        ):
            mock_cls.return_value = mock_policy
            wrapper = ACTWrapper(act_cfg, device="cpu")

        obs = {"observation.state": np.zeros(3, dtype=np.float32)}
        action = wrapper.select_action(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (8,)
        assert action.dtype == np.float32

    def test_forward_returns_scalar_loss_tensor(self, act_cfg: Any) -> None:
        from playground.policies.act_wrapper import ACTWrapper  # noqa: PLC0415

        mock_policy = _make_mock_lerobot_policy()
        with (
            patch("playground.policies.act_wrapper.ACTPolicy") as mock_cls,
            patch("playground.policies.act_wrapper.ACTConfig"),
            patch("playground.policies.act_wrapper._LEROBOT_AVAILABLE", True),
        ):
            mock_cls.return_value = mock_policy
            wrapper = ACTWrapper(act_cfg, device="cpu")

        batch = {
            "observation.state": torch.zeros(4, 1, 3),
            "action": torch.zeros(4, 100, 8),
        }
        loss = wrapper.forward(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # scalar
        assert float(loss) == pytest.approx(0.42)

    def test_reset_delegates_to_policy(self, act_cfg: Any) -> None:
        from playground.policies.act_wrapper import ACTWrapper  # noqa: PLC0415

        mock_policy = _make_mock_lerobot_policy()
        with (
            patch("playground.policies.act_wrapper.ACTPolicy") as mock_cls,
            patch("playground.policies.act_wrapper.ACTConfig"),
            patch("playground.policies.act_wrapper._LEROBOT_AVAILABLE", True),
        ):
            mock_cls.return_value = mock_policy
            wrapper = ACTWrapper(act_cfg, device="cpu")

        wrapper.reset()
        mock_policy.reset.assert_called_once()

    def test_save_calls_torch_save(self, act_cfg: Any, tmp_path: Path) -> None:
        from playground.policies.act_wrapper import ACTWrapper  # noqa: PLC0415

        mock_policy = _make_mock_lerobot_policy()
        ckpt_path = tmp_path / "checkpoint_00000100.ckpt"

        with (
            patch("playground.policies.act_wrapper.ACTPolicy") as mock_cls,
            patch("playground.policies.act_wrapper.ACTConfig"),
            patch("playground.policies.act_wrapper._LEROBOT_AVAILABLE", True),
            patch("playground.policies.act_wrapper.torch") as mock_torch,
        ):
            mock_cls.return_value = mock_policy
            wrapper = ACTWrapper(act_cfg, device="cpu")
            wrapper.save(ckpt_path)
            mock_torch.save.assert_called_once()

    def test_load_checkpoint_restores_state_dict(self, act_cfg: Any, tmp_path: Path) -> None:
        from playground.policies.act_wrapper import ACTWrapper  # noqa: PLC0415

        mock_policy = _make_mock_lerobot_policy()
        ckpt_path = tmp_path / "checkpoint_00000100.ckpt"
        fake_state = {"w": torch.zeros(4)}

        with (
            patch("playground.policies.act_wrapper.ACTPolicy") as mock_cls,
            patch("playground.policies.act_wrapper.ACTConfig"),
            patch("playground.policies.act_wrapper._LEROBOT_AVAILABLE", True),
            patch("playground.policies.act_wrapper.torch") as mock_torch,
        ):
            mock_cls.return_value = mock_policy
            mock_torch.load.return_value = {"model": fake_state}
            mock_torch.device.return_value = "cpu"
            wrapper = ACTWrapper(act_cfg, device="cpu")
            wrapper.load_checkpoint(ckpt_path)

        mock_policy.load_state_dict.assert_called_once()
        called_with = mock_policy.load_state_dict.call_args[0][0]
        assert called_with is fake_state

    def test_parameters_exposes_policy_params(self, act_cfg: Any) -> None:
        from playground.policies.act_wrapper import ACTWrapper  # noqa: PLC0415

        mock_policy = _make_mock_lerobot_policy()
        param = torch.nn.Parameter(torch.zeros(3))
        mock_policy.parameters.return_value = iter([param])

        with (
            patch("playground.policies.act_wrapper.ACTPolicy") as mock_cls,
            patch("playground.policies.act_wrapper.ACTConfig"),
            patch("playground.policies.act_wrapper._LEROBOT_AVAILABLE", True),
        ):
            mock_cls.return_value = mock_policy
            wrapper = ACTWrapper(act_cfg, device="cpu")

        params = list(wrapper.parameters())
        assert len(params) == 1
        assert params[0] is param


# ---------------------------------------------------------------------------
# DiffusionWrapper tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDiffusionWrapper:
    """Unit tests for DiffusionWrapper — lerobot DiffusionPolicy is fully mocked."""

    def test_fresh_init_builds_policy_from_config(self, diffusion_cfg: Any) -> None:
        from playground.policies.diffusion_wrapper import DiffusionWrapper  # noqa: PLC0415

        mock_policy = _make_mock_lerobot_policy()
        with (
            patch("playground.policies.diffusion_wrapper.DiffusionPolicy") as mock_cls,
            patch("playground.policies.diffusion_wrapper.DiffusionConfig") as mock_cfg_cls,
            patch("playground.policies.diffusion_wrapper._LEROBOT_AVAILABLE", True),
        ):
            mock_cls.return_value = mock_policy
            DiffusionWrapper(diffusion_cfg, device="cpu")
            mock_cfg_cls.assert_called_once()
            mock_cls.assert_called_once()

    def test_pretrained_init_calls_from_pretrained(self, diffusion_cfg_pretrained: Any) -> None:
        from playground.policies.diffusion_wrapper import DiffusionWrapper  # noqa: PLC0415

        mock_policy = _make_mock_lerobot_policy()
        with (
            patch("playground.policies.diffusion_wrapper.DiffusionPolicy") as mock_cls,
            patch("playground.policies.diffusion_wrapper._LEROBOT_AVAILABLE", True),
        ):
            mock_cls.from_pretrained.return_value = mock_policy
            DiffusionWrapper(diffusion_cfg_pretrained, device="cpu")
            mock_cls.from_pretrained.assert_called_once_with(
                "mefiezvous/cube-reach-v1-diffusion"
            )

    def test_select_action_returns_numpy_of_correct_shape(self, diffusion_cfg: Any) -> None:
        from playground.policies.diffusion_wrapper import DiffusionWrapper  # noqa: PLC0415

        mock_policy = _make_mock_lerobot_policy()
        mock_policy.select_action.return_value = torch.zeros(1, 8)

        with (
            patch("playground.policies.diffusion_wrapper.DiffusionPolicy") as mock_cls,
            patch("playground.policies.diffusion_wrapper.DiffusionConfig"),
            patch("playground.policies.diffusion_wrapper._LEROBOT_AVAILABLE", True),
        ):
            mock_cls.return_value = mock_policy
            wrapper = DiffusionWrapper(diffusion_cfg, device="cpu")

        obs = {"observation.state": np.zeros(3, dtype=np.float32)}
        action = wrapper.select_action(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (8,)

    def test_forward_returns_scalar_loss_tensor(self, diffusion_cfg: Any) -> None:
        from playground.policies.diffusion_wrapper import DiffusionWrapper  # noqa: PLC0415

        mock_policy = _make_mock_lerobot_policy()
        with (
            patch("playground.policies.diffusion_wrapper.DiffusionPolicy") as mock_cls,
            patch("playground.policies.diffusion_wrapper.DiffusionConfig"),
            patch("playground.policies.diffusion_wrapper._LEROBOT_AVAILABLE", True),
        ):
            mock_cls.return_value = mock_policy
            wrapper = DiffusionWrapper(diffusion_cfg, device="cpu")

        batch = {
            "observation.state": torch.zeros(4, 2, 3),
            "action": torch.zeros(4, 8, 8),
        }
        loss = wrapper.forward(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()

    def test_reset_delegates_to_policy(self, diffusion_cfg: Any) -> None:
        from playground.policies.diffusion_wrapper import DiffusionWrapper  # noqa: PLC0415

        mock_policy = _make_mock_lerobot_policy()
        with (
            patch("playground.policies.diffusion_wrapper.DiffusionPolicy") as mock_cls,
            patch("playground.policies.diffusion_wrapper.DiffusionConfig"),
            patch("playground.policies.diffusion_wrapper._LEROBOT_AVAILABLE", True),
        ):
            mock_cls.return_value = mock_policy
            wrapper = DiffusionWrapper(diffusion_cfg, device="cpu")

        wrapper.reset()
        mock_policy.reset.assert_called_once()
