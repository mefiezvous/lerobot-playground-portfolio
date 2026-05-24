# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Episode collection entrypoint — Hydra-configurable CLI.

Collects an episode dataset for an arbitrary robot using
``mlcore.collection.*`` primitives. Replaces the legacy hardcoded
``DemoCollector`` in :mod:`playground.data.pipeline`.

Usage::

    # Default config (scripted policy, 200 episodes)
    python collect.py

    # Quick smoke (2 episodes, custom root)
    python collect.py episodes=2 dataset.root=/tmp/smoke

    # Push to the Hub at the end
    python collect.py push_to_hub=true
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from loguru import logger
from mlcore.collection.scripted import ScriptedCollector, ScriptedReachPolicy
from mlcore.collection.sink import HubSink
from mlcore.observation.builder import ObservationBuilder
from mlcore.robots import get as get_robot_spec
from mlcore.robots import validate_spec_against_env
from omegaconf import DictConfig, OmegaConf
from robotics_platform.envs.registry import EnvAdapterRegistry

import playground.envs.registrations  # type: ignore[import-untyped]  # noqa: F401 — populates EnvAdapterRegistry


def _env_cfg(cfg: DictConfig) -> DictConfig:
    """Return the env sub-config, handling the legacy double-wrap.

    The existing ``configs/env/cube_reach_v1.yaml`` declares
    ``# @package _global_`` *below* the SPDX header. Hydra only honours the
    directive when it is the literal first line, so the env contents end
    up nested under ``cfg.env.env`` instead of ``cfg.env``. We accept both
    shapes so this CLI keeps working if the env yaml is later fixed.
    """
    env = cfg.env
    if "name" not in env and "env" in env:
        return env.env  # type: ignore[no-any-return]
    return env  # type: ignore[no-any-return]


@hydra.main(config_path="configs", config_name="collect/default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main collection entry point.

    Args:
        cfg: Composed Hydra config (env + dataset + collect overrides).
    """
    episodes_n: int = int(cfg.get("episodes", 200))
    policy_type: str = str(cfg.get("policy_type", "scripted"))
    push_to_hub: bool = bool(cfg.get("push_to_hub", False))
    seed: int = int(cfg.get("seed", 42))

    env_cfg = _env_cfg(cfg)

    logger.info(
        "Starting collection | env={} episodes={} policy={} push_to_hub={}",
        env_cfg.name,
        episodes_n,
        policy_type,
        push_to_hub,
    )
    logger.debug("Full config:\n{}", OmegaConf.to_yaml(cfg))

    # Build env + validate spec ↔ env (fail-fast).
    env_name = str(env_cfg.name)
    logger.info("Validating robot spec ↔ env for '{}'...", env_name)
    env_cls = EnvAdapterRegistry.get(env_name)
    env = env_cls()
    try:
        spec = get_robot_spec(env_name)
        validate_spec_against_env(spec, env)
        logger.info("Spec validation OK")

        # Build collector based on policy_type.
        if policy_type == "scripted":
            policy = ScriptedReachPolicy(
                action_dim=spec.action_dim,
                ee_key=spec.ee_pos_key or "ee_pos",
                target_key=spec.target_pos_key,
                seed=seed,
            )
            collector = ScriptedCollector(
                policy,
                task_id=spec.name,
                task_description=str(env_cfg.get("task_description", spec.name)),
                seed=seed,
            )
        elif policy_type == "teleop":
            # ``TeleopCollector`` is currently a stub (see
            # ``mlcore.collection.teleop``); its real Collector contract
            # is not finalised, so we refuse to fabricate an API here.
            raise NotImplementedError(
                "teleop branch awaits TeleopCollector finalisation — see ROADMAP"
            )
        else:
            raise ValueError(
                f"Unknown policy_type {policy_type!r} — expected 'scripted' or 'teleop'"
            )

        # Sample one obs to resolve obs_builder shapes from the spec.
        sample_obs, _ = env.reset(seed=seed)
        obs_builder = ObservationBuilder.from_spec(spec, sample_obs)

        # ScriptedCollector.collect closes the env when it is done.
        episodes = collector.collect(env, n_episodes=episodes_n)
    except BaseException:
        # Make sure the env is closed on any failure before reset/collect.
        try:
            env.close()
        except Exception as exc:  # pragma: no cover — defensive cleanup
            logger.debug("env.close() during cleanup raised: {}", exc)
        raise

    # Persist to LeRobotDataset v3.0 (and optionally push to Hub).
    root = Path(cfg.dataset.root)
    root.mkdir(parents=True, exist_ok=True)
    sink = HubSink(
        repo_id=cfg.dataset.repo_id,
        fps=int(env_cfg.fps),
        action_dim=spec.action_dim,
    )
    sink.write(
        episodes,
        root=root,
        obs_builder=obs_builder,
        push_to_hub=push_to_hub,
    )

    logger.info(
        "Collection done | episodes={} root={} repo_id={}",
        len(episodes),
        root,
        cfg.dataset.repo_id,
    )


def _entrypoint() -> Any:
    """Console-script entrypoint wrapper (Hydra owns argv)."""
    return main()


if __name__ == "__main__":
    main()
