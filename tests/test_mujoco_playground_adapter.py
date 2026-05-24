# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for MujocoPlaygroundAdapter and its registry entry.

All tests mock the underlying mujoco_playground env so they run on any
platform without GPU or JAX CUDA support.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from robotics_platform.envs.interfaces import EnvAdapter
from robotics_platform.envs.registry import EnvAdapterRegistry

# Import side-effect — registers the cube_reach_v1 adapter under
# "cube_reach_v1" in the EnvAdapterRegistry.
import playground.envs.registrations  # noqa: F401
from playground.envs.mujoco_playground_adapter import MujocoPlaygroundAdapter


def _make_obs(ee_x: float = 0.3, cube_x: float = 0.0) -> dict[str, Any]:
    return {
        "ee_pos": np.array([ee_x, 0.0, 0.5], dtype=np.float32),
        "cube_pos": np.array([cube_x, 0.0, 0.5], dtype=np.float32),
        "joint_positions": np.zeros(7, dtype=np.float32),
    }


def _make_underlying_env() -> MagicMock:
    """Fake mujoco_playground env returning gym-style 5-tuples."""
    env = MagicMock()
    env.reset.return_value = (_make_obs(), {"info_key": "info_val"})
    env.step.return_value = (_make_obs(), 0.5, False, False, {"success": False})
    env.action_size = 8
    return env


@pytest.mark.unit
class TestMujocoPlaygroundAdapterContract:
    """Verify the adapter implements the EnvAdapter Protocol."""

    def test_instance_is_envadapter(self) -> None:
        adapter = MujocoPlaygroundAdapter(env_name="CubeReachV1", task_description="Reach the cube")
        assert isinstance(adapter, EnvAdapter)

    def test_task_description_returned_unchanged(self) -> None:
        adapter = MujocoPlaygroundAdapter(env_name="CubeReachV1", task_description="Reach the cube")
        assert adapter.task_description == "Reach the cube"


@pytest.mark.unit
class TestMujocoPlaygroundAdapterDelegation:
    """Verify reset/step/close delegate to the lazily-loaded underlying env."""

    def test_reset_delegates_to_underlying_env(self) -> None:
        underlying = _make_underlying_env()
        adapter = MujocoPlaygroundAdapter(env_name="CubeReachV1", task_description="t")
        with patch(
            "playground.envs.mujoco_playground_adapter._load_mp_env",
            return_value=underlying,
        ):
            obs, info = adapter.reset(seed=7)

        underlying.reset.assert_called_once_with(seed=7)
        assert "ee_pos" in obs
        assert info == {"info_key": "info_val"}

    def test_step_delegates_and_returns_5_tuple(self) -> None:
        underlying = _make_underlying_env()
        adapter = MujocoPlaygroundAdapter(env_name="CubeReachV1", task_description="t")
        with patch(
            "playground.envs.mujoco_playground_adapter._load_mp_env",
            return_value=underlying,
        ):
            adapter.reset(seed=0)
            action = np.zeros(8, dtype=np.float32)
            obs, reward, terminated, truncated, info = adapter.step(action)

        underlying.step.assert_called_once()
        np.testing.assert_array_equal(underlying.step.call_args[0][0], action)
        assert reward == 0.5
        assert terminated is False
        assert truncated is False
        assert info == {"success": False}
        assert "ee_pos" in obs

    def test_close_delegates_to_underlying(self) -> None:
        underlying = _make_underlying_env()
        adapter = MujocoPlaygroundAdapter(env_name="CubeReachV1", task_description="t")
        with patch(
            "playground.envs.mujoco_playground_adapter._load_mp_env",
            return_value=underlying,
        ):
            adapter.reset(seed=0)
            adapter.close()

        underlying.close.assert_called_once()

    def test_close_before_reset_is_noop(self) -> None:
        adapter = MujocoPlaygroundAdapter(env_name="CubeReachV1", task_description="t")
        # Should not raise when underlying env never resolved.
        adapter.close()

    def test_underlying_loaded_only_once(self) -> None:
        """Lazy resolution must cache the env across multiple resets."""
        underlying = _make_underlying_env()
        adapter = MujocoPlaygroundAdapter(env_name="CubeReachV1", task_description="t")
        with patch(
            "playground.envs.mujoco_playground_adapter._load_mp_env",
            return_value=underlying,
        ) as load_mock:
            adapter.reset(seed=0)
            adapter.reset(seed=1)
            adapter.step(np.zeros(8, dtype=np.float32))

        assert load_mock.call_count == 1


@pytest.mark.unit
class TestMujocoPlaygroundAdapterObsKeysAndActionDim:
    def test_obs_space_keys_sorted_from_first_reset(self) -> None:
        underlying = _make_underlying_env()
        adapter = MujocoPlaygroundAdapter(env_name="CubeReachV1", task_description="t")
        with patch(
            "playground.envs.mujoco_playground_adapter._load_mp_env",
            return_value=underlying,
        ):
            adapter.reset(seed=0)
            keys = adapter.obs_space_keys

        assert keys == sorted(["ee_pos", "cube_pos", "joint_positions"])

    def test_obs_space_keys_triggers_reset_if_not_called_yet(self) -> None:
        underlying = _make_underlying_env()
        adapter = MujocoPlaygroundAdapter(env_name="CubeReachV1", task_description="t")
        with patch(
            "playground.envs.mujoco_playground_adapter._load_mp_env",
            return_value=underlying,
        ):
            keys = adapter.obs_space_keys

        assert "ee_pos" in keys

    def test_action_dim_reads_action_size_from_underlying(self) -> None:
        underlying = _make_underlying_env()
        underlying.action_size = 8
        adapter = MujocoPlaygroundAdapter(env_name="CubeReachV1", task_description="t")
        with patch(
            "playground.envs.mujoco_playground_adapter._load_mp_env",
            return_value=underlying,
        ):
            assert adapter.action_dim == 8


@pytest.mark.unit
class TestMujocoPlaygroundAdapterRegistry:
    def test_cube_reach_v1_registered_under_namespaced_name(self) -> None:
        factory = EnvAdapterRegistry.get("cube_reach_v1")
        # Factory must be callable with no args and return an EnvAdapter.
        instance = factory()
        assert isinstance(instance, EnvAdapter)

    def test_registered_factory_returns_mujoco_playground_adapter(self) -> None:
        factory = EnvAdapterRegistry.get("cube_reach_v1")
        instance = factory()
        assert isinstance(instance, MujocoPlaygroundAdapter)

    def test_registered_factory_has_expected_task_description(self) -> None:
        factory = EnvAdapterRegistry.get("cube_reach_v1")
        instance = factory()
        assert instance.task_description == "Reach the cube"
