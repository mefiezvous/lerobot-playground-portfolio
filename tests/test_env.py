# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for CubeReachV1 custom environment.

Unit tests cover the pure reward function (_reach_reward) and require only
JAX on CPU — no GPU, no real mujoco_playground env.

Integration tests exercise mp.make("CubeReachV1") and require a working
mujoco_playground installation (real or numpy-backend).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest

# conftest.py ensures this import succeeds even when mujoco_playground is mocked.
from playground.envs.cube_reach_v1 import CubeReachV1, _reach_reward


@pytest.mark.unit
class TestReachReward:
    """Pure-function reward tests — no GPU or real env required."""

    def test_reward_decreases_with_distance(self) -> None:
        cube = jnp.zeros(3)
        near = _reach_reward(jnp.array([0.02, 0.0, 0.0]), cube)
        far = _reach_reward(jnp.array([0.5, 0.0, 0.0]), cube)
        assert float(near) > float(far)

    def test_reward_near_cube_exceeds_0_9(self) -> None:
        # dist ≈ 0.01 → reach_reward ≈ 1 - tanh(0.05) ≈ 0.95
        reward = _reach_reward(jnp.array([0.01, 0.0, 0.0]), jnp.zeros(3))
        assert float(reward) > 0.9

    def test_success_bonus_below_threshold(self) -> None:
        # dist = 0.04 < SUCCESS_THRESHOLD (0.05) → bonus +5.0
        reward = _reach_reward(jnp.array([0.04, 0.0, 0.0]), jnp.zeros(3))
        assert float(reward) > 5.0

    def test_no_success_bonus_above_threshold(self) -> None:
        # dist = 0.1 > SUCCESS_THRESHOLD → no bonus
        reward = _reach_reward(jnp.array([0.1, 0.0, 0.0]), jnp.zeros(3))
        assert float(reward) < 5.0

    def test_success_threshold_at_boundary(self) -> None:
        # dist exactly at threshold — boundary is exclusive (dist < threshold)
        reward_inside = _reach_reward(
            jnp.array([0.049, 0.0, 0.0]), jnp.zeros(3), threshold=0.05
        )
        reward_outside = _reach_reward(
            jnp.array([0.051, 0.0, 0.0]), jnp.zeros(3), threshold=0.05
        )
        assert float(reward_inside) > 5.0
        assert float(reward_outside) < 5.0

    def test_class_success_threshold_constant(self) -> None:
        assert CubeReachV1.SUCCESS_THRESHOLD == pytest.approx(0.05)

    def test_class_compute_reward_delegates_to_reach_reward(self) -> None:
        # object.__new__ bypasses PandaPickCube.__init__; safe because
        # compute_reward only calls _reach_reward (no env state needed).
        env = object.__new__(CubeReachV1)
        ee = jnp.array([0.03, 0.0, 0.0])
        cube = jnp.zeros(3)
        class_result = env.compute_reward({"ee_pos": ee, "cube_pos": cube}, {})
        fn_result = _reach_reward(ee, cube, CubeReachV1.SUCCESS_THRESHOLD)
        assert float(class_result) == pytest.approx(float(fn_result))


@pytest.mark.integration
class TestCubeReachV1Integration:
    """Full-env tests — skipped when mujoco_playground is absent or mocked."""

    @pytest.fixture(autouse=True)
    def _require_real_mp(self) -> None:
        mp = pytest.importorskip(
            "mujoco_playground", reason="mujoco_playground not installed"
        )
        # Skip if conftest installed a mock (MagicMock is not a real package).
        if isinstance(mp.make, type(MagicMock())):  # type: ignore[call-overload]
            pytest.skip("mujoco_playground is mocked — integration test skipped")

    def test_env_registered_and_makeable(self) -> None:
        import mujoco_playground as mp

        import playground.envs.cube_reach_v1  # noqa: F401 — registers env

        env = mp.make("CubeReachV1")
        assert env is not None

    def test_reset_returns_dict_with_required_keys(self) -> None:
        import mujoco_playground as mp

        env = mp.make("CubeReachV1")
        obs, _ = env.reset()
        assert "ee_pos" in obs
        assert "cube_pos" in obs

    def test_step_returns_five_tuple(self) -> None:
        import mujoco_playground as mp

        env = mp.make("CubeReachV1")
        env.reset()
        result = env.step(jnp.zeros(8))
        assert len(result) == 5  # obs, reward, terminated, truncated, info

    def test_episode_truncates_at_200_steps(self) -> None:
        import mujoco_playground as mp

        env = mp.make("CubeReachV1")
        env.reset()
        action = jnp.zeros(8)
        done = False
        steps = 0
        while not done and steps < 250:
            _, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
        assert steps <= 200
