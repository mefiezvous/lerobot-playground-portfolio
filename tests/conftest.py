# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Shared test configuration.

Installs a minimal mujoco_playground mock when the real package is absent
so that unit tests (reward logic, pipeline data structures) can run on any
platform without GPU or JAX CUDA support.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock


def _install_mp_mock() -> None:
    """Patch sys.modules with a lightweight mujoco_playground stub."""

    class _StubPandaPickCube:
        """Minimal base class — gives CubeReachV1 a valid parent."""

    pick_mod = MagicMock()
    pick_mod.PandaPickCube = _StubPandaPickCube
    pick_mod.default_config = MagicMock(return_value=MagicMock())

    panda_pkg = MagicMock()
    panda_pkg.pick = pick_mod

    manipulation_mod = MagicMock()
    manipulation_mod.register_environment = MagicMock()

    src_mod = MagicMock()
    src_mod.manipulation = manipulation_mod

    mp_stub = MagicMock()
    mp_stub._src = src_mod
    mp_stub.registry = MagicMock()

    sys.modules.setdefault("mujoco_playground", mp_stub)
    sys.modules.setdefault("mujoco_playground._src", src_mod)
    sys.modules.setdefault("mujoco_playground._src.manipulation", manipulation_mod)
    sys.modules.setdefault("mujoco_playground._src.manipulation.franka_emika_panda", panda_pkg)
    sys.modules.setdefault("mujoco_playground._src.manipulation.franka_emika_panda.pick", pick_mod)


try:
    import mujoco_playground as _real_mp  # noqa: F401
except (ImportError, OSError):
    _install_mp_mock()
