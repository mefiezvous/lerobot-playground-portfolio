# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""EnvAdapter registrations for the playground custom envs.

Importing this module has the side-effect of populating the
:class:`robotics_platform.envs.registry.EnvAdapterRegistry` with the
playground-specific entries. Each registered name maps to a zero-arg
factory (a thin :class:`MujocoPlaygroundAdapter` subclass) that the
pipeline can instantiate without knowing about the underlying simulator.

In addition, every ``robot_specs/*.yaml`` entry is loaded as a
:class:`mlcore.robots.base.RobotSpec` and — for ``adapter.type:
mujoco_playground`` entries — registered as a factory too. YAML
registrations run after the static ones above, so a YAML spec sharing an
``id`` with a hardcoded adapter wins (with a logged warning).
"""

from __future__ import annotations

import os
from pathlib import Path

from mlcore.robots.yaml_loader import load_specs_from_dir
from robotics_platform.envs.registry import register

from playground.envs.mujoco_playground_adapter import MujocoPlaygroundAdapter
from playground.envs.yaml_registrations import register_from_yaml


@register("cube_reach_v1")
class CubeReachV1Adapter(MujocoPlaygroundAdapter):
    """Zero-arg adapter for the CubeReachV1 MuJoCo Playground env."""

    def __init__(self) -> None:
        super().__init__(
            env_name="CubeReachV1",
            task_description="Reach the cube",
        )


# Module-level bootstrap: registrations.py is imported purely for its
# side effects (see playground.data.pipeline), before Hydra config is
# composed, so the specs directory is resolved via env var / repo-relative
# default rather than Hydra.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROBOT_SPECS_DIR = Path(os.environ.get("ROBOT_SPECS_DIR", str(_REPO_ROOT / "robot_specs")))

load_specs_from_dir(_ROBOT_SPECS_DIR)
register_from_yaml(_ROBOT_SPECS_DIR)
