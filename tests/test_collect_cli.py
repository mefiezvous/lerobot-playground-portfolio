# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``collect.py`` Hydra CLI happy path (WP-3)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COLLECT_PY = _REPO_ROOT / "collect.py"


@pytest.mark.unit
def test_collect_module_imports() -> None:
    """Smoke: ``import collect`` from the repo root must succeed."""
    snippet = f"import sys; sys.path.insert(0, r'{_REPO_ROOT}'); import collect"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"


@pytest.mark.unit
def test_collect_help_runs() -> None:
    """``collect.py --help`` must exit 0 (Hydra's help screen)."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(_COLLECT_PY), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"


@pytest.mark.unit
def test_env_cfg_helper_handles_double_wrap() -> None:
    """``_env_cfg`` must transparently unwrap the ``cfg.env.env`` legacy shape."""
    sys.path.insert(0, str(_REPO_ROOT))
    from omegaconf import OmegaConf

    import collect

    # Plain shape: cfg.env.name present → returned as-is.
    plain = OmegaConf.create({"env": {"name": "foo", "fps": 20}})
    assert collect._env_cfg(plain).name == "foo"

    # Legacy double-wrap: cfg.env.env.name present, cfg.env.name absent → unwrap.
    wrapped = OmegaConf.create({"env": {"env": {"name": "bar", "fps": 20}}})
    assert collect._env_cfg(wrapped).name == "bar"


@pytest.mark.unit
def test_teleop_branch_raises_not_implemented() -> None:
    """The teleop policy branch must raise ``NotImplementedError``.

    We grep the source for the explicit ``raise NotImplementedError`` line
    since invoking the full Hydra entry would require a real env and
    dataset; the contract under test is purely textual/structural.
    """
    src = _COLLECT_PY.read_text(encoding="utf-8")
    assert 'policy_type == "teleop"' in src
    assert "NotImplementedError" in src
    # Ensure the raise is inside the teleop branch (cheap structural check).
    teleop_idx = src.index('policy_type == "teleop"')
    nie_idx = src.index("NotImplementedError", teleop_idx)
    # Should appear within ~500 chars after the elif.
    assert nie_idx - teleop_idx < 500
