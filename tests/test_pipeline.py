# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for the demo data pipeline.

All tests are pure unit tests: mujoco_playground and LeRobotDataset are mocked
so no GPU or network access is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from playground.data.pipeline import DemoCollector, Episode, ScriptedPolicy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_obs(ee_x: float = 0.3, cube_x: float = 0.0) -> dict[str, Any]:
    return {
        "ee_pos": np.array([ee_x, 0.0, 0.5], dtype=np.float32),
        "cube_pos": np.array([cube_x, 0.0, 0.5], dtype=np.float32),
        "joint_positions": np.zeros(7, dtype=np.float32),
    }


def _make_mp_env(n_steps_until_done: int = 10) -> MagicMock:
    """Fake mujoco_playground env that terminates after n steps."""
    env = MagicMock()
    env.reset.return_value = (_make_obs(), {})

    call_count: list[int] = [0]

    def _step(action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        call_count[0] += 1
        done = call_count[0] >= n_steps_until_done
        return _make_obs(), 0.5, done, False, {"success": done}

    env.step.side_effect = _step
    return env


# ---------------------------------------------------------------------------
# ScriptedPolicy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScriptedPolicy:
    def test_select_action_returns_array_of_shape_8(self) -> None:
        policy = ScriptedPolicy(gain=2.0)
        action = policy.select_action(_make_obs())
        assert action.shape == (8,)
        assert action.dtype == np.float32

    def test_action_moves_toward_cube(self) -> None:
        # ee is at x=0.3, cube at x=0.0 → delta_x = -0.3 → action[0] < 0
        policy = ScriptedPolicy(gain=2.0, noise_scale=0.0)
        obs = _make_obs(ee_x=0.3, cube_x=0.0)
        action = policy.select_action(obs)
        assert action[0] < 0.0

    def test_action_clipped_to_minus_one_one(self) -> None:
        policy = ScriptedPolicy(gain=100.0, noise_scale=0.0)
        action = policy.select_action(_make_obs(ee_x=5.0))
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_deterministic_with_zero_noise(self) -> None:
        policy = ScriptedPolicy(gain=2.0, noise_scale=0.0)
        obs = _make_obs()
        a1 = policy.select_action(obs)
        a2 = policy.select_action(obs)
        np.testing.assert_array_equal(a1, a2)


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEpisode:
    def test_episode_stores_fields(self) -> None:
        ep = Episode(
            observations=[_make_obs()],
            actions=[np.zeros(8, dtype=np.float32)],
            rewards=[0.5],
            success=True,
        )
        assert len(ep.observations) == 1
        assert len(ep.actions) == 1
        assert ep.success is True


# ---------------------------------------------------------------------------
# DemoCollector.collect
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDemoCollector:
    def test_collect_returns_correct_episode_count(self) -> None:
        env = _make_mp_env(n_steps_until_done=5)
        policy = ScriptedPolicy(gain=2.0, noise_scale=0.0)
        collector = DemoCollector(policy=policy, fps=20)

        with patch("playground.data.pipeline.mp") as mp_mock:
            mp_mock.make.return_value = env
            episodes = collector.collect(env_name="CubeReachV1", n_episodes=3)

        assert len(episodes) == 3

    def test_collect_episode_has_correct_step_count(self) -> None:
        n_steps = 7
        env = _make_mp_env(n_steps_until_done=n_steps)
        policy = ScriptedPolicy(gain=2.0, noise_scale=0.0)
        collector = DemoCollector(policy=policy, fps=20)

        with patch("playground.data.pipeline.mp") as mp_mock:
            mp_mock.make.return_value = env
            episodes = collector.collect(env_name="CubeReachV1", n_episodes=1)

        ep = episodes[0]
        assert len(ep.observations) == n_steps
        assert len(ep.actions) == n_steps
        assert len(ep.rewards) == n_steps

    def test_collect_sets_success_flag_from_info(self) -> None:
        env = _make_mp_env(n_steps_until_done=3)
        policy = ScriptedPolicy(gain=2.0, noise_scale=0.0)
        collector = DemoCollector(policy=policy, fps=20)

        with patch("playground.data.pipeline.mp") as mp_mock:
            mp_mock.make.return_value = env
            episodes = collector.collect(env_name="CubeReachV1", n_episodes=1)

        # Last step info has success=True (from _make_mp_env)
        assert episodes[0].success is True

    # ------------------------------------------------------------------
    # to_lerobot_dataset
    # ------------------------------------------------------------------

    def test_to_lerobot_dataset_calls_create(self, tmp_path: Path) -> None:
        episodes = [
            Episode(
                observations=[_make_obs()] * 5,
                actions=[np.zeros(8, dtype=np.float32)] * 5,
                rewards=[0.1] * 5,
                success=False,
            )
            for _ in range(3)
        ]
        policy = ScriptedPolicy()
        collector = DemoCollector(policy=policy, fps=20)

        mock_dataset = MagicMock()
        with patch("playground.data.pipeline.LeRobotDataset") as lrd_mock:
            lrd_mock.create.return_value = mock_dataset
            collector.to_lerobot_dataset(
                episodes=episodes,
                repo_id="mefiezvous/test-dataset",
                root=tmp_path,
            )

        lrd_mock.create.assert_called_once()
        call_kwargs = lrd_mock.create.call_args.kwargs
        assert call_kwargs["repo_id"] == "mefiezvous/test-dataset"
        assert call_kwargs["fps"] == 20

    def test_to_lerobot_dataset_adds_correct_frame_count(self, tmp_path: Path) -> None:
        n_episodes, steps_per_ep = 2, 4
        episodes = [
            Episode(
                observations=[_make_obs()] * steps_per_ep,
                actions=[np.zeros(8, dtype=np.float32)] * steps_per_ep,
                rewards=[0.2] * steps_per_ep,
                success=True,
            )
            for _ in range(n_episodes)
        ]
        policy = ScriptedPolicy()
        collector = DemoCollector(policy=policy, fps=20)

        mock_dataset = MagicMock()
        with patch("playground.data.pipeline.LeRobotDataset") as lrd_mock:
            lrd_mock.create.return_value = mock_dataset
            collector.to_lerobot_dataset(
                episodes=episodes,
                repo_id="mefiezvous/test-dataset",
                root=tmp_path,
            )

        assert mock_dataset.add_frame.call_count == n_episodes * steps_per_ep
        assert mock_dataset.save_episode.call_count == n_episodes

    def test_to_lerobot_dataset_features_include_state_and_action(self, tmp_path: Path) -> None:
        episodes = [
            Episode(
                observations=[_make_obs()],
                actions=[np.zeros(8, dtype=np.float32)],
                rewards=[0.0],
                success=False,
            )
        ]
        policy = ScriptedPolicy()
        collector = DemoCollector(policy=policy, fps=20)

        mock_dataset = MagicMock()
        with patch("playground.data.pipeline.LeRobotDataset") as lrd_mock:
            lrd_mock.create.return_value = mock_dataset
            collector.to_lerobot_dataset(
                episodes=episodes,
                repo_id="mefiezvous/test-dataset",
                root=tmp_path,
            )

        features = lrd_mock.create.call_args.kwargs["features"]
        assert "observation.state" in features
        assert "action" in features
