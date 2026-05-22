# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for playground.utils.visualizer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from playground.utils.visualizer import EpisodeFrames, plot_rewards, render_episode


def _episode(**kw: object) -> EpisodeFrames:
    defaults: dict[str, object] = {
        "frames": [np.zeros((64, 64, 3), dtype=np.uint8)] * 5,
        "rewards": [1.0, 2.0, 3.0, 2.0, 1.0],
        "robot_name": "cube_reach_v1",
        "policy_type": "act",
        "success": True,
    }
    defaults.update(kw)
    return EpisodeFrames(**defaults)  # type: ignore[arg-type]


def _patch_imageio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out _imageio inside visualizer so no real ffmpeg is needed."""
    mock_imageio = MagicMock()
    mock_imageio.mimwrite.side_effect = lambda path, frames, fps=30: Path(path).touch()
    monkeypatch.setattr("playground.utils.visualizer._imageio", mock_imageio)
    monkeypatch.setattr("playground.utils.visualizer._IMAGEIO_AVAILABLE", True)


@pytest.mark.unit
def test_render_episode_returns_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_imageio(monkeypatch)
    result = render_episode(_episode(), tmp_path)
    assert result is not None
    assert result.exists()


@pytest.mark.unit
def test_render_episode_creates_mp4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_imageio(monkeypatch)
    result = render_episode(_episode(), tmp_path)
    assert result is not None
    assert result.suffix == ".mp4"


@pytest.mark.unit
def test_render_episode_filename_includes_robot_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_imageio(monkeypatch)
    result = render_episode(_episode(robot_name="my_robot_arm"), tmp_path)
    assert result is not None
    assert "my_robot_arm" in result.name


@pytest.mark.unit
def test_plot_rewards_creates_png(tmp_path: Path) -> None:
    result = plot_rewards(_episode(), tmp_path)
    assert result is not None
    assert result.suffix == ".png"


@pytest.mark.unit
def test_plot_rewards_values(tmp_path: Path) -> None:
    result = plot_rewards(_episode(), tmp_path)
    assert result is not None
    assert result.exists()
    assert result.stat().st_size > 1024
