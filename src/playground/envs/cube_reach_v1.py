# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""CubeReachV1 — dense reach-only reward environment for the Franka Panda."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from mujoco_playground import register  # type: ignore[import-untyped]
from mujoco_playground._src.manipulation.panda_pick_cube import (  # type: ignore[import-untyped]
    PandaPickCube,
)


def _reach_reward(
    ee_pos: jax.Array,
    cube_pos: jax.Array,
    threshold: float = 0.05,
) -> jax.Array:
    """Compute dense reach reward between end-effector and cube.

    Args:
        ee_pos: End-effector XYZ position, shape (3,).
        cube_pos: Cube XYZ position, shape (3,).
        threshold: Success distance in metres.

    Returns:
        Scalar reward: reach_reward ∈ [0, 1] plus 5.0 success bonus.
    """
    dist: jax.Array = jnp.linalg.norm(ee_pos - cube_pos)
    reach_reward: jax.Array = 1.0 - jnp.tanh(5.0 * dist)
    success: jax.Array = dist < threshold
    return reach_reward + 5.0 * success.astype(jnp.float32)


class CubeReachV1(PandaPickCube):
    """PandaPickCube subclass with dense reach-only reward.

    Overrides only the reward signal. Physics, observations, and episode
    length (200 steps) are inherited from PandaPickCube.

    Reward structure:
      - reach_reward = 1 − tanh(5 · dist_ee_cube)  ∈ [0, 1]
      - +5.0 success bonus when dist < SUCCESS_THRESHOLD (0.05 m)
    """

    SUCCESS_THRESHOLD: float = 0.05  # metres

    def compute_reward(
        self,
        obs: dict[str, Any],
        info: dict[str, Any],
    ) -> jax.Array:
        """Dense reach reward — delegates to module-level _reach_reward."""
        return _reach_reward(
            obs["ee_pos"],
            obs["cube_pos"],
            self.SUCCESS_THRESHOLD,
        )


register("CubeReachV1", CubeReachV1)
