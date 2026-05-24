# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""MujocoPlaygroundAdapter — wrap a MuJoCo Playground env as an EnvAdapter.

Bridges the gap between the third-party ``mujoco_playground`` registry and the
generic :class:`robotics_platform.envs.interfaces.EnvAdapter` Protocol, so the
training and data pipelines can consume any underlying simulator backend
through a single contract.

The underlying env is resolved lazily on the first call to ``reset`` (or any
property that needs it). This keeps the adapter constructible in test
environments where ``mujoco_playground`` is mocked or absent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _load_mp_env(env_name: str) -> Any:
    """Resolve a MuJoCo Playground env by name.

    Isolated as a module-level function so tests can patch it without
    needing to import the real ``mujoco_playground`` package.

    Args:
        env_name: Registered mujoco_playground env name (e.g. ``"CubeReachV1"``).

    Returns:
        The underlying env instance returned by ``mp.registry.load``.

    Raises:
        RuntimeError: If ``mujoco_playground`` cannot be imported.
    """
    try:
        import mujoco_playground as mp
    except (ImportError, OSError) as exc:  # pragma: no cover - env-specific
        raise RuntimeError(
            "mujoco_playground is required for MujocoPlaygroundAdapter. Install with: uv sync"
        ) from exc
    return mp.registry.load(env_name)


class MujocoPlaygroundAdapter:
    """Adapter wrapping a MuJoCo Playground env in the EnvAdapter Protocol.

    The underlying env is loaded lazily on first use, allowing the adapter
    to be instantiated (and registered) in environments where the heavy
    mujoco_playground stack is absent — typically unit tests.

    Args:
        env_name: Name registered with ``mujoco_playground.registry``.
        task_description: Natural-language task instruction surfaced to
            LLM-conditioned policies via :attr:`task_description`.
    """

    def __init__(self, env_name: str, task_description: str) -> None:
        self._env_name = env_name
        self._task_description = task_description
        self._env: Any = None
        self._obs_keys: list[str] | None = None

    # ------------------------------------------------------------------
    # Lazy resolution
    # ------------------------------------------------------------------
    def _ensure_env(self) -> Any:
        if self._env is None:
            self._env = _load_mp_env(self._env_name)
        return self._env

    # ------------------------------------------------------------------
    # EnvAdapter Protocol
    # ------------------------------------------------------------------
    def reset(self, seed: int) -> tuple[dict[str, NDArray[np.floating[Any]]], dict[str, Any]]:
        """Reset the underlying env and cache obs keys for introspection."""
        env = self._ensure_env()
        obs, info = env.reset(seed=seed)
        if self._obs_keys is None:
            self._obs_keys = sorted(obs.keys())
        return obs, info

    def step(
        self, action: NDArray[np.floating[Any]]
    ) -> tuple[
        dict[str, NDArray[np.floating[Any]]],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Delegate to the underlying env's gym-style 5-tuple step."""
        env = self._ensure_env()
        obs, reward, terminated, truncated, info = env.step(action)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        """Release the underlying env (no-op when never resolved)."""
        if self._env is not None:
            self._env.close()

    @property
    def obs_space_keys(self) -> list[str]:
        """Sorted list of observation dict keys.

        Triggers a one-time ``reset(seed=0)`` if no reset has occurred yet,
        so callers can introspect the obs schema before kicking off rollouts.
        """
        if self._obs_keys is None:
            self.reset(seed=0)
        assert self._obs_keys is not None
        return self._obs_keys

    @property
    def action_dim(self) -> int:
        """Dimension of the action vector accepted by ``step``.

        Reads ``action_size`` from the underlying env (mujoco_playground
        convention).
        """
        env = self._ensure_env()
        return int(env.action_size)

    @property
    def task_description(self) -> str:
        """Constructor-supplied task instruction string."""
        return self._task_description
