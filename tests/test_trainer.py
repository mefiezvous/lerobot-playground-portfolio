# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for Trainer — mock env + mock policy, no GPU or real lerobot required.

Mock strategy
-------------
- ``policy``: a MagicMock that implements the ACTWrapper/DiffusionWrapper interface
  (forward, reset, save, load_checkpoint, parameters).
- ``dataloader``: a plain Python iterable of small dummy batches.
- ``mlflow``: patched at the trainer's module level.
- ``os.environ``: patched per test for WandB gate checks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def training_cfg() -> Any:
    return OmegaConf.create(
        {
            "dataset": {"repo_id": "mefiezvous/cube-reach-v1-dataset", "root": None},
            "policy": {"name": "act"},
            "training": {
                "total_steps": 6,
                "save_every_steps": 3,
                "checkpoint_dir": "checkpoints/",
                "batch_size": 4,
                "num_envs": 1,
                "device": "cpu",
                "mixed_precision": False,
                "seed": 42,
                "lr": 1e-4,
            },
            "logging": {
                "backend": "mlflow",
                "mlflow_tracking_uri": "mlruns/",
                "experiment_name": "test_experiment",
                "run_name": "test_run",
            },
        }
    )


def _make_mock_policy() -> MagicMock:
    """MagicMock that looks like an ACTWrapper / DiffusionWrapper."""
    policy = MagicMock()
    policy.forward.return_value = torch.tensor(0.5, requires_grad=True)
    param = torch.nn.Parameter(torch.zeros(4))
    policy.parameters.return_value = iter([param])
    return policy


def _make_dataloader(n_batches: int = 10) -> list[dict[str, torch.Tensor]]:
    """Return a simple list-based "dataloader" with dummy batches."""
    return [
        {
            "observation.state": torch.zeros(4, 1, 3),
            "action": torch.zeros(4, 100, 8),
        }
        for _ in range(n_batches)
    ]


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrainer:
    """Unit tests for Trainer — all external I/O is mocked."""

    def _make_trainer(self, cfg: Any, policy: MagicMock, dataloader: Any) -> Any:
        """Instantiate a Trainer with mlflow fully mocked."""
        from playground.training.trainer import Trainer  # noqa: PLC0415

        with patch("playground.training.trainer.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            trainer = Trainer(cfg, policy, dataloader)
        return trainer

    def test_init_calls_mlflow_set_tracking_uri(self, training_cfg: Any) -> None:
        from playground.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        dataloader = _make_dataloader()

        with patch("playground.training.trainer.mlflow") as mock_mlflow:
            Trainer(training_cfg, policy, dataloader)

        mock_mlflow.set_tracking_uri.assert_called_once_with("mlruns/")

    def test_init_starts_mlflow_experiment(self, training_cfg: Any) -> None:
        from playground.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        dataloader = _make_dataloader()

        with patch("playground.training.trainer.mlflow") as mock_mlflow:
            Trainer(training_cfg, policy, dataloader)

        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")

    def test_detect_no_checkpoint_returns_none(
        self, training_cfg: Any, tmp_path: Path
    ) -> None:
        from playground.training.trainer import Trainer  # noqa: PLC0415

        cfg = OmegaConf.merge(
            training_cfg,
            {"training": {"checkpoint_dir": str(tmp_path / "empty_ckpts")}},
        )
        policy = _make_mock_policy()

        with patch("playground.training.trainer.mlflow"):
            trainer = Trainer(cfg, policy, _make_dataloader())

        assert trainer._detect_latest_checkpoint() is None

    def test_detect_checkpoint_finds_latest(
        self, training_cfg: Any, tmp_path: Path
    ) -> None:
        from playground.training.trainer import Trainer  # noqa: PLC0415

        # Create fake checkpoint files
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint_00001000.ckpt").touch()
        (ckpt_dir / "checkpoint_00002000.ckpt").touch()
        (ckpt_dir / "checkpoint_00003000.ckpt").touch()

        cfg = OmegaConf.merge(
            training_cfg, {"training": {"checkpoint_dir": str(ckpt_dir)}}
        )
        policy = _make_mock_policy()

        with patch("playground.training.trainer.mlflow"):
            trainer = Trainer(cfg, policy, _make_dataloader())

        latest = trainer._detect_latest_checkpoint()
        assert latest is not None
        assert latest.name == "checkpoint_00003000.ckpt"

    def test_resume_from_checkpoint_updates_start_step(
        self, training_cfg: Any, tmp_path: Path
    ) -> None:
        from playground.training.trainer import Trainer  # noqa: PLC0415

        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        ckpt_file = ckpt_dir / "checkpoint_00005000.ckpt"
        ckpt_file.touch()

        cfg = OmegaConf.merge(
            training_cfg,
            {
                "training": {
                    "checkpoint_dir": str(ckpt_dir),
                    "total_steps": 10000,
                    "save_every_steps": 1000,
                }
            },
        )
        policy = _make_mock_policy()

        with patch("playground.training.trainer.mlflow"):
            trainer = Trainer(cfg, policy, _make_dataloader())

        assert trainer._step == 5000
        policy.load_checkpoint.assert_called_once_with(ckpt_file)

    def test_train_calls_forward_total_steps_times(self, training_cfg: Any) -> None:
        from playground.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        dataloader = _make_dataloader(n_batches=20)

        with (
            patch("playground.training.trainer.mlflow"),
            patch("playground.training.trainer.torch") as mock_torch,
        ):
            # Let Adam work normally; mock only torch.device
            mock_torch.device.return_value = torch.device("cpu")
            mock_torch.optim = torch.optim
            trainer = Trainer(training_cfg, policy, dataloader)
            trainer.train()

        # total_steps=6 → forward called 6 times
        assert policy.forward.call_count == 6

    def test_train_saves_checkpoint_at_save_every_steps(
        self, training_cfg: Any, tmp_path: Path
    ) -> None:
        from playground.training.trainer import Trainer  # noqa: PLC0415

        cfg = OmegaConf.merge(
            training_cfg, {"training": {"checkpoint_dir": str(tmp_path / "ckpts")}}
        )
        policy = _make_mock_policy()
        dataloader = _make_dataloader(n_batches=20)

        with patch("playground.training.trainer.mlflow"):
            trainer = Trainer(cfg, policy, dataloader)
            trainer.train()

        # save_every_steps=3, total_steps=6 → 2 saves (at step 3 and step 6)
        assert policy.save.call_count == 2

    def test_train_logs_mlflow_at_save_every_steps(self, training_cfg: Any) -> None:
        from playground.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        dataloader = _make_dataloader(n_batches=20)

        with patch("playground.training.trainer.mlflow") as mock_mlflow:
            trainer = Trainer(training_cfg, policy, dataloader)
            trainer.train()

        # save_every_steps=3, total_steps=6 → 2 mlflow log_metrics calls
        assert mock_mlflow.log_metrics.call_count == 2
        # Verify metric keys on first call
        first_call_kwargs = mock_mlflow.log_metrics.call_args_list[0]
        logged_metrics = first_call_kwargs[0][0]  # first positional arg
        assert "loss" in logged_metrics
        assert "reward" in logged_metrics
        assert "success_rate" in logged_metrics

    def test_wandb_disabled_without_api_key(self, training_cfg: Any) -> None:
        """_setup_wandb sets _wandb_enabled=False when WANDB_API_KEY is absent."""
        from playground.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()

        with (
            patch("playground.training.trainer.mlflow"),
            patch.dict(os.environ, {}, clear=True),  # no WANDB_API_KEY
        ):
            trainer = Trainer(training_cfg, policy, _make_dataloader())
            trainer._setup_wandb()

        assert trainer._wandb_enabled is False

    def test_train_dataloader_cycles_when_exhausted(self, training_cfg: Any) -> None:
        from playground.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        # dataloader shorter than total_steps → must cycle
        short_dataloader = _make_dataloader(n_batches=2)

        with patch("playground.training.trainer.mlflow"):
            trainer = Trainer(training_cfg, policy, short_dataloader)
            # Should not raise StopIteration even though dataloader has only 2 batches
            trainer.train()

        assert policy.forward.call_count == 6
