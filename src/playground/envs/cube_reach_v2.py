# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Stub environment for cube_reach_v2.

TODO:
  1. Choose a MuJoCo Playground base env (e.g. PandaPickCube).
  2. Subclass it and override compute_reward().
  3. Register: mp.register("CubeReachV2", CubeReachV2)
"""

from __future__ import annotations

# from mujoco_playground._src.manipulation.panda_pick_cube import PandaPickCube
# from mujoco_playground import register
# from typing import Any
# import jax
#
# class CubeReachV2(PandaPickCube):
#     def compute_reward(self, obs: dict[str, Any], info: dict[str, Any]) -> jax.Array:
#         raise NotImplementedError
#
# register("CubeReachV2", CubeReachV2)
