# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Demo data pipeline: scripted policy → episode collection → LeRobotDataset v3.0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    import mujoco_playground as mp
except (ImportError, OSError):
    mp = None  # type: ignore[assignment]

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    LeRobotDataset = None  # type: ignore[assignment]


@dataclass
class Episode:
    """Single collected trajectory.

    Attributes:
        observations: Raw obs dicts from mujoco_playground (one per step).
        actions: Actions applied at each step, shape (8,) each.
        rewards: Scalar reward received at each step.
        success: True if the episode ended with a successful reach.
    """

    observations: list[dict[str, Any]]
    actions: list[np.ndarray[Any, Any]]
    rewards: list[float]
    success: bool = False


class ScriptedPolicy:
    """Proportional reach controller for scripted demo collection.

    Moves the end-effector toward the cube with a simple P-controller
    in Cartesian space, padded to the full 8-DOF action vector.

    Args:
        gain: Proportional gain applied to the EE–cube delta.
        noise_scale: Std-dev of Gaussian noise added to the action.
        seed: RNG seed for reproducible noise.
    """

    def __init__(
        self,
        gain: float = 2.0,
        noise_scale: float = 0.05,
        seed: int = 42,
    ) -> None:
        self._gain = gain
        self._noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)

    def select_action(self, obs: dict[str, Any]) -> np.ndarray[Any, Any]:
        """Return an 8-D action that moves the EE toward the cube.

        Args:
            obs: Raw observation dict with keys ``ee_pos`` and ``cube_pos``.

        Returns:
            Action array of shape (8,) and dtype float32, clipped to [-1, 1].
        """
        ee_pos = np.asarray(obs.get("ee_pos", np.zeros(3)), dtype=np.float32)
        cube_pos = np.asarray(obs.get("cube_pos", np.zeros(3)), dtype=np.float32)
        delta = cube_pos - ee_pos

        action = np.zeros(8, dtype=np.float32)
        action[:3] = self._gain * delta[:3]
        if self._noise_scale > 0.0:
            noise = self._rng.normal(0.0, self._noise_scale, 8).astype(np.float32)
            action += noise
        return np.clip(action, -1.0, 1.0)


class DemoCollector:
    """Collect scripted demonstration episodes and export to LeRobotDataset v3.0.

    Args:
        policy: Scripted policy used to generate actions.
        fps: Control frequency used to tag the LeRobotDataset.
        seed: Base random seed; episode i uses seed + i.
    """

    def __init__(
        self,
        policy: ScriptedPolicy,
        fps: int = 20,
        seed: int = 42,
    ) -> None:
        self._policy = policy
        self._fps = fps
        self._seed = seed

    def collect(self, env_name: str, n_episodes: int) -> list[Episode]:
        """Run the scripted policy for n_episodes and return collected data.

        Args:
            env_name: Registered mujoco_playground environment name.
            n_episodes: Number of episodes to collect.

        Returns:
            List of Episode objects, one per episode.
        """
        if mp is None:
            raise RuntimeError(
                "mujoco_playground is required for demo collection. "
                "Install with: uv sync"
            )

        env = mp.make(env_name)  # type: ignore[attr-defined]
        episodes: list[Episode] = []

        for i in range(n_episodes):
            obs, _ = env.reset(seed=self._seed + i)
            ep_obs: list[dict[str, Any]] = []
            ep_actions: list[np.ndarray[Any, Any]] = []
            ep_rewards: list[float] = []
            success = False
            done = False

            while not done:
                action = self._policy.select_action(obs)
                ep_obs.append(obs)
                ep_actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_rewards.append(float(reward))
                done = bool(terminated or truncated)
                success = bool(info.get("success", False))

            episodes.append(
                Episode(
                    observations=ep_obs,
                    actions=ep_actions,
                    rewards=ep_rewards,
                    success=success,
                )
            )
            logger.info(
                f"Episode {i + 1}/{n_episodes} done | "
                f"steps={len(ep_rewards)} | success={success}"
            )

        env.close()
        return episodes

    def to_lerobot_dataset(
        self,
        episodes: list[Episode],
        repo_id: str,
        root: Path,
    ) -> Any:
        """Export collected episodes to a LeRobotDataset v3.0 on disk.

        Args:
            episodes: Episodes returned by :meth:`collect`.
            repo_id: HuggingFace dataset repo id (e.g. ``mefiezvous/cube-reach-v1-dataset``).
            root: Local directory where the dataset is written.

        Returns:
            The created ``LeRobotDataset`` instance.
        """
        if LeRobotDataset is None:
            raise RuntimeError(
                "lerobot is required for dataset export. Install with: uv sync"
            )

        features: dict[str, Any] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["ee_x", "ee_y", "ee_z"],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": None,
            },
        }

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=self._fps,
            root=root,
            features=features,
        )

        total_frames = 0
        for ep_idx, episode in enumerate(episodes):
            for obs, action in zip(episode.observations, episode.actions, strict=False):
                dataset.add_frame(
                    {
                        "observation.state": np.asarray(
                            obs.get("ee_pos", np.zeros(3)), dtype=np.float32
                        ),
                        "action": np.asarray(action, dtype=np.float32),
                    }
                )
                total_frames += 1
            dataset.save_episode(task="Reach the cube")
            logger.debug(f"Saved episode {ep_idx + 1}/{len(episodes)}")

        logger.info(
            f"Dataset saved to {root} | episodes={len(episodes)} | frames={total_frames}"
        )
        return dataset
