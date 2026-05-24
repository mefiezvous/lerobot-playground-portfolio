# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Demo data pipeline: scripted policy → episode collection → LeRobotDataset v3.0.

DEPRECATED — use ``mlcore.collection.*`` via the root-level ``collect.py``
Hydra CLI instead. ``DemoCollector`` and ``ScriptedPolicy`` are kept only
because legacy tests still import them; both will be removed once the
migration to ``collect.py`` is complete.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from mlcore.data.schema import (
    ACTION_KEY,
    OBS_STATE_KEY,
    SUCCESS_KEY,
    TASK_DESCRIPTION_KEY,
    TASK_ID_KEY,
    build_features,
)
from mlcore.observation.builder import ObservationBuilder
from robotics_platform.envs.registry import EnvAdapterRegistry

# Import side-effect: registers playground env adapters in the registry.
import playground.envs.registrations  # noqa: F401
from playground.envs.mujoco_playground_adapter import MujocoPlaygroundAdapter


def _default_obs_builder() -> ObservationBuilder:
    """Return the canonical 16-dim ObservationBuilder for the CubeReach tasks.

    Layout: ee_pos(3) + cube_pos(3) + joint_positions(7) + (cube_pos - ee_pos)(3) = 16.
    """
    return ObservationBuilder(
        keys=("ee_pos", "cube_pos", "joint_positions"),
        key_shapes=(3, 3, 7),
        relational=(("cube_pos", "ee_pos"),),
    )


def _resolve_env(env_name: str) -> Any:
    """Return an EnvAdapter for ``env_name``.

    Resolution order:

    1. If ``env_name`` is registered in :class:`EnvAdapterRegistry`, instantiate
       the registered factory (zero-arg constructor expected).
    2. Otherwise — backward-compat fallback — wrap the name in a
       :class:`MujocoPlaygroundAdapter` so legacy bare names like
       ``"CubeReachV1"`` keep working without an explicit registration.

    Args:
        env_name: Either a namespaced registry key (e.g.
            ``"cube_reach_v1"``) or a raw mujoco_playground env name.

    Returns:
        An :class:`~robotics_platform.envs.interfaces.EnvAdapter` instance.
    """
    try:
        factory = EnvAdapterRegistry.get(env_name)
    except KeyError:
        logger.warning(
            f"env '{env_name}' not in EnvAdapterRegistry — falling back to "
            "MujocoPlaygroundAdapter wrapping. Consider registering it explicitly."
        )
        return MujocoPlaygroundAdapter(env_name=env_name, task_description="")
    return factory()


try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    LeRobotDataset = None


@dataclass
class Episode:
    """Single collected trajectory.

    Attributes:
        observations: Raw obs dicts from the env (one per step).
        actions: Actions applied at each step, shape (8,) each.
        rewards: Scalar reward received at each step.
        task_id: Short, stable identifier for the task (e.g. ``"cube_reach_v1"``).
        task_description: Natural-language description of the task, used both as
            a per-frame label and as the ``task`` arg to ``save_episode``.
        success: True if the episode ended with a successful reach.
    """

    observations: list[dict[str, Any]]
    actions: list[np.ndarray[Any, Any]]
    rewards: list[float]
    task_id: str
    task_description: str
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
        obs_builder: Optional :class:`ObservationBuilder` that flattens raw obs
            dicts into the LeRobotDataset state vector. Defaults to the
            canonical 16-dim CubeReach layout.
        task_id: Short, stable identifier stamped on every collected episode.
            Defaults to ``"cube_reach_v1"`` for the current CubeReach use case.
        task_description: Natural-language task label stamped on every collected
            episode and forwarded as the ``task`` argument to
            ``LeRobotDataset.save_episode``. Defaults to ``"Reach the red cube"``.
    """

    def __init__(
        self,
        policy: ScriptedPolicy,
        fps: int = 20,
        seed: int = 42,
        obs_builder: ObservationBuilder | None = None,
        task_id: str = "cube_reach_v1",
        task_description: str = "Reach the red cube",
    ) -> None:
        warnings.warn(
            "DemoCollector is deprecated — use mlcore.collection.* via the "
            "root-level collect.py Hydra CLI. This class will be removed "
            "after the migration is complete.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._policy = policy
        self._fps = fps
        self._seed = seed
        self._obs_builder = obs_builder if obs_builder is not None else _default_obs_builder()
        self._task_id = task_id
        self._task_description = task_description

    def collect(self, env_name: str, n_episodes: int) -> list[Episode]:
        """Run the scripted policy for n_episodes and return collected data.

        Args:
            env_name: EnvAdapter registry key (e.g.
                ``"cube_reach_v1"``) or legacy bare mujoco_playground
                env name (auto-wrapped as a fallback).
            n_episodes: Number of episodes to collect.

        Returns:
            List of Episode objects, one per episode.
        """
        env: Any = _resolve_env(env_name)
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
                    task_id=self._task_id,
                    task_description=self._task_description,
                    success=success,
                )
            )
            logger.info(
                f"Episode {i + 1}/{n_episodes} done | steps={len(ep_rewards)} | success={success}"
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
            raise RuntimeError("lerobot is required for dataset export. Install with: uv sync")

        builder = self._obs_builder
        features: dict[str, Any] = build_features(
            state_dim=builder.state_dim,
            state_names=builder.state_names,
            action_dim=8,
        )

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=self._fps,
            root=root,
            features=features,
        )

        total_frames = 0
        for ep_idx, episode in enumerate(episodes):
            success_arr = np.array([episode.success], dtype=np.bool_)
            for obs, action in zip(episode.observations, episode.actions, strict=False):
                dataset.add_frame(
                    {
                        OBS_STATE_KEY: builder.build(obs),
                        ACTION_KEY: np.asarray(action, dtype=np.float32),
                        TASK_ID_KEY: episode.task_id,
                        TASK_DESCRIPTION_KEY: episode.task_description,
                        SUCCESS_KEY: success_arr,
                    }
                )
                total_frames += 1
            dataset.save_episode(task=episode.task_description)
            logger.debug(f"Saved episode {ep_idx + 1}/{len(episodes)}")

        logger.info(f"Dataset saved to {root} | episodes={len(episodes)} | frames={total_frames}")
        return dataset
