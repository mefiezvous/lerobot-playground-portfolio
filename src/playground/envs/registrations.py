# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""EnvAdapter registrations for the playground custom envs.

Importing this module has the side-effect of populating the
:class:`robotics_platform.envs.registry.EnvAdapterRegistry` with the
playground-specific entries.

Every ``robot_specs/*.yaml`` entry is loaded as a
:class:`mlcore.robots.base.RobotSpec` and — for ``adapter.type:
mujoco_playground`` entries — registered as a zero-arg
:class:`MujocoPlaygroundAdapter` factory (a ``functools.partial``) so the
pipeline can instantiate it without knowing about the underlying
simulator. Hardware/real-robot adapters (no ``adapter`` block, or an
unsupported ``adapter.type``) remain code-only and would be registered
here via an explicit ``@register`` decorator.
"""

from __future__ import annotations

import os
from pathlib import Path

from mlcore.robots.yaml_loader import load_specs_from_dir

from playground.envs.yaml_registrations import register_from_yaml

# Module-level bootstrap: registrations.py is imported purely for its
# side effects (see playground.data.pipeline), before Hydra config is
# composed, so the specs directory is resolved via env var / repo-relative
# default rather than Hydra.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROBOT_SPECS_DIR = Path(os.environ.get("ROBOT_SPECS_DIR", str(_REPO_ROOT / "robot_specs")))

load_specs_from_dir(_ROBOT_SPECS_DIR)
register_from_yaml(_ROBOT_SPECS_DIR)
