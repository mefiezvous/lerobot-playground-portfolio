# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""EnvAdapter registrations for the playground custom envs.

Importing this module has the side-effect of populating the
:class:`robotics_platform.envs.registry.EnvAdapterRegistry` with the
playground-specific entries. Each registered name maps to a zero-arg
factory (a thin :class:`MujocoPlaygroundAdapter` subclass) that the
pipeline can instantiate without knowing about the underlying simulator.
"""

from __future__ import annotations

from robotics_platform.envs.registry import register

from playground.envs.mujoco_playground_adapter import MujocoPlaygroundAdapter


@register("mujoco_pgnd:cube_reach_v1")
class CubeReachV1Adapter(MujocoPlaygroundAdapter):
    """Zero-arg adapter for the CubeReachV1 MuJoCo Playground env."""

    def __init__(self) -> None:
        super().__init__(
            env_name="CubeReachV1",
            task_description="Reach the cube",
        )
