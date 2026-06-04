# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``playground.scripts.add_robot`` scaffolding CLI.

These tests exercise the script on the real workspace tree (because the
target paths are hardcoded relative to ``__file__``). We use a sentinel
robot name and rigorously clean up all artifacts in a ``try/finally``
block so the workspace is left untouched.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

# Sentinel name unlikely to collide with anything real.
# Must match _NAME_RE = r"^[a-z][a-z0-9_]{1,63}$" (LRB-004 validation).
_SENTINEL = "pytest_smoke_robot"


def _module() -> object:
    """Import add_robot fresh so we can re-patch sys.argv between calls."""
    import playground.scripts.add_robot as mod

    importlib.reload(mod)
    return mod


def _expected_paths() -> dict[str, Path]:
    mod = _module()
    return {
        "robot_cfg": mod._ROBOT_CFG_DIR / f"{_SENTINEL}.yaml",  # type: ignore[attr-defined]
        "env_stub": mod._ENVS_DIR / f"{_SENTINEL}.py",  # type: ignore[attr-defined]
        "spec": mod._MLCORE_SPECS_DIR / f"{_SENTINEL}.py",  # type: ignore[attr-defined]
        "env_yaml": mod._HYDRA_ENV_DIR / f"{_SENTINEL}.yaml",  # type: ignore[attr-defined]
        "dataset_yaml": mod._HYDRA_DATASET_DIR / f"{_SENTINEL}.yaml",  # type: ignore[attr-defined]
    }


def _registrations_file() -> Path:
    mod = _module()
    return mod._REGISTRATIONS_FILE  # type: ignore[attr-defined,no-any-return]


@pytest.fixture
def cleanup_workspace() -> Iterator[None]:
    """Snapshot registrations.py + delete sentinel files before and after."""
    paths = _expected_paths()
    reg_file = _registrations_file()
    original_reg = reg_file.read_text(encoding="utf-8") if reg_file.exists() else ""

    # Pre-clean in case of a previous failed test.
    for p in paths.values():
        if p.exists():
            p.unlink()
    if reg_file.exists():
        reg_file.write_text(original_reg, encoding="utf-8")

    try:
        yield
    finally:
        for p in paths.values():
            if p.exists():
                p.unlink()
        # Restore registrations.py to its original content.
        if reg_file.exists() or original_reg:
            reg_file.write_text(original_reg, encoding="utf-8")


def _invoke(args: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Invoke add_robot.main() with patched sys.argv."""
    monkeypatch.setattr(sys, "argv", ["add_robot", *args])
    mod = _module()
    mod.main()  # type: ignore[attr-defined]


@pytest.mark.unit
def test_dry_run_writes_nothing(monkeypatch: pytest.MonkeyPatch, cleanup_workspace: None) -> None:
    paths = _expected_paths()
    _invoke(
        [
            _SENTINEL,
            "--n-joints",
            "6",
            "--action-dim",
            "6",
            "--obs-keys",
            "ee_pos,target_pos",
            "--dry-run",
        ],
        monkeypatch,
    )
    for label, p in paths.items():
        assert not p.exists(), f"dry-run should not create {label}: {p}"


@pytest.mark.unit
def test_full_scaffold_generates_5_files(
    monkeypatch: pytest.MonkeyPatch, cleanup_workspace: None
) -> None:
    paths = _expected_paths()
    reg_file = _registrations_file()
    before = reg_file.read_text(encoding="utf-8") if reg_file.exists() else ""

    _invoke(
        [
            _SENTINEL,
            "--n-joints",
            "6",
            "--action-dim",
            "6",
            "--obs-keys",
            "ee_pos,target_pos",
            "--task-description",
            "smoke test",
        ],
        monkeypatch,
    )

    for label, p in paths.items():
        assert p.exists(), f"missing scaffolded file {label}: {p}"

    after = reg_file.read_text(encoding="utf-8")
    assert f'@register("mujoco_pgnd:{_SENTINEL}")' in after
    assert len(after) > len(before)


@pytest.mark.unit
def test_idempotent_rerun_skips_existing(
    monkeypatch: pytest.MonkeyPatch, cleanup_workspace: None, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [
        _SENTINEL,
        "--n-joints",
        "6",
        "--action-dim",
        "6",
        "--obs-keys",
        "ee_pos,target_pos",
    ]
    _invoke(args, monkeypatch)
    # Second run should not raise.
    _invoke(args, monkeypatch)

    # registrations.py must not contain a duplicate registration.
    reg_text = _registrations_file().read_text(encoding="utf-8")
    assert reg_text.count(f'@register("mujoco_pgnd:{_SENTINEL}")') == 1


@pytest.mark.unit
def test_script_module_is_importable() -> None:
    """Smoke test: importing the module must not error."""
    result = subprocess.run(
        [sys.executable, "-c", "import playground.scripts.add_robot"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
