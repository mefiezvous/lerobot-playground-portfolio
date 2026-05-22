# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Episode visualisation utilities: MP4 rendering and reward plotting."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    import imageio as _imageio

    _IMAGEIO_AVAILABLE = True
except ImportError:
    _imageio = None  # type: ignore[assignment]
    _IMAGEIO_AVAILABLE = False


@dataclass
class EpisodeFrames:
    frames: list[np.ndarray[Any, Any]]  # (H, W, 3) uint8
    rewards: list[float]
    robot_name: str
    policy_type: str
    success: bool


def render_episode(episode: EpisodeFrames, output_dir: Path) -> Path | None:
    """Encode frames as .mp4 via imageio[ffmpeg].

    Returns None with a warning if imageio is unavailable or frames list is empty.
    Filename: {robot_name}_{policy_type}_{timestamp}.mp4
    """
    if not _IMAGEIO_AVAILABLE:
        logger.warning("imageio not installed — skipping video render. Run: pip install imageio[ffmpeg]")  # noqa: E501
        return None
    if not episode.frames:
        logger.warning("No frames in episode — skipping video render.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    path = output_dir / f"{episode.robot_name}_{episode.policy_type}_{ts}.mp4"
    _imageio.mimwrite(str(path), episode.frames, fps=30)  # type: ignore[arg-type]
    logger.info(f"Rendered episode video → {path}")
    return path


def plot_rewards(episode: EpisodeFrames, output_dir: Path | None = None) -> Path:
    """Plot reward-per-step curve, saved as .png.

    Uses the Agg (non-interactive) backend so it runs in headless environments.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = output_dir or Path(".")
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    path = out / f"{episode.robot_name}_{episode.policy_type}_{ts}_rewards.png"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episode.rewards, marker="o", linewidth=1.5, markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title(
        f"{episode.robot_name} / {episode.policy_type}"
        f" — {'success' if episode.success else 'failure'}"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=100)
    plt.close(fig)

    logger.info(f"Saved reward plot → {path}")
    return path
