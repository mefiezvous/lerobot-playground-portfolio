# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for eval.py entrypoint. Evaluator unit tests live in ml-core."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf


def _make_mock_policy() -> MagicMock:
    policy = MagicMock()
    policy.select_action.return_value = np.zeros(8, dtype=np.float32)
    return policy


@pytest.mark.unit
def test_eval_entrypoint_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import eval as eval_module  # noqa: PLC0415

    assert callable(eval_module.main)

    monkeypatch.chdir(tmp_path)

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
