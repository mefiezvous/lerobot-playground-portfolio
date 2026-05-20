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

    panda_mod = MagicMock()
    panda_mod.PandaPickCube = _StubPandaPickCube

    mp_stub = MagicMock()
    mp_stub.register = MagicMock()
    mp_stub._src = MagicMock()
    mp_stub._src.manipulation = MagicMock()
    mp_stub._src.manipulation.panda_pick_cube = panda_mod
    mp_stub._src.manipulation.panda_pick_cube.PandaPickCube = _StubPandaPickCube

    sys.modules.setdefault("mujoco_playground", mp_stub)
    sys.modules.setdefault("mujoco_playground._src", mp_stub._src)
    sys.modules.setdefault("mujoco_playground._src.manipulation", mp_stub._src.manipulation)
    sys.modules.setdefault(
        "mujoco_playground._src.manipulation.panda_pick_cube", panda_mod
    )


try:
    import mujoco_playground as _real_mp  # noqa: F401
except (ImportError, OSError):
    _install_mp_mock()
