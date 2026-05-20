# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for Evaluator and eval.py entrypoint.

Mock strategy
-------------
- ``policy``: MagicMock with select_action returning a fixed numpy array.
- ``env``: MagicMock whose step/reset simulate one-step episodes.
- ``mlflow``: patched at the evaluator's module level.
- File I/O in _run_eval: handled by monkeypatch.chdir(tmp_path).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_policy() -> MagicMock:
    policy = MagicMock()
    policy.select_action.return_value = np.zeros(8, dtype=np.float32)
    return policy


def _make_mock_env(*, success: bool = False) -> MagicMock:
    """Mock env whose step() terminates immediately with one step."""
    env = MagicMock()
    env.reset.return_value = np.zeros(3, dtype=np.float32)
    env.step.return_value = (np.zeros(3), 1.0, True, {"success": success})
    return env


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvaluator:
    def test_evaluator_runs_n_episodes(self) -> None:
        from playground.eval.evaluator import Evaluator  # noqa: PLC0415

        policy = _make_mock_policy()
        env = _make_mock_env()
        ev = Evaluator(policy, "CubeReachV1", n_episodes=5, env=env)
        result = ev.evaluate()

        assert result.n_episodes == 5
        assert env.reset.call_count == 5
        assert len(result.per_episode_rewards) == 5

    def test_evaluator_success_rate(self) -> None:
        from playground.eval.evaluator import EvalResult, Evaluator  # noqa: PLC0415

        policy = _make_mock_policy()
        env: MagicMock = MagicMock()
        env.reset.return_value = np.zeros(3)
        # 2 successes, 1 failure — each episode terminates in one step
        env.step.side_effect = [
            (np.zeros(3), 5.0, True, {"success": True}),
            (np.zeros(3), 5.0, True, {"success": True}),
            (np.zeros(3), 0.5, True, {"success": False}),
        ]

        ev = Evaluator(policy, "CubeReachV1", n_episodes=3, env=env)
        result = ev.evaluate()

        assert isinstance(result, EvalResult)
        assert result.success_rate == pytest.approx(2 / 3)

    def test_evaluator_logs_to_mlflow(self) -> None:
        from playground.eval.evaluator import Evaluator  # noqa: PLC0415

        policy = _make_mock_policy()
        env = _make_mock_env(success=True)

        with patch("playground.eval.evaluator.mlflow") as mock_mlflow:
            ev = Evaluator(
                policy, "CubeReachV1", n_episodes=2, mlflow_run_id="run_abc", env=env
            )
            ev.evaluate()

        mock_mlflow.start_run.assert_called_once_with(run_id="run_abc")
        mock_mlflow.log_metrics.assert_called_once()
        logged: Any = mock_mlflow.log_metrics.call_args[0][0]
        assert "eval/success_rate" in logged
        assert "eval/mean_reward" in logged

    def test_eval_entrypoint_smoke(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import eval as eval_module  # noqa: PLC0415

        assert callable(eval_module.main)

        monkeypatch.chdir(tmp_path)  # eval_report.json is written here

        policy = _make_mock_policy()
        mock_result = MagicMock()
        mock_result.n_episodes = 2
        mock_result.success_rate = 1.0
        mock_result.mean_reward = 5.0
        mock_result.per_episode_rewards = [5.0, 5.0]

        cfg = OmegaConf.create(
            {
                "env": {"name": "CubeReachV1"},
                "policy": {"name": "act"},
                "training": {"device": "cpu", "checkpoint_dir": "checkpoints/"},
                "eval": {
                    "checkpoint_path": str(tmp_path / "checkpoint_00001000.ckpt"),
                    "n_episodes": 2,
                    "mlflow_run_id": None,
                },
            }
        )

        with (
            patch("eval.ACTWrapper") as MockACT,
            patch("eval.Evaluator") as MockEvaluator,
        ):
            MockACT.return_value = policy
            MockEvaluator.return_value.evaluate.return_value = mock_result

            eval_module._run_eval(cfg)

        MockEvaluator.assert_called_once()
        MockEvaluator.return_value.evaluate.assert_called_once()
