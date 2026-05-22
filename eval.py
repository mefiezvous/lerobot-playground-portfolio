# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Evaluation entrypoint — Hydra-configurable CLI.

Usage::

    # Evaluate an ACT checkpoint (50 episodes by default)
    python eval.py +eval.checkpoint_path=checkpoints/checkpoint_00010000.ckpt \\
        +eval.n_episodes=50 logging.run_name=eval_run_001

    # Diffusion checkpoint, log back into an existing MLflow run
    python eval.py policy=diffusion \\
        +eval.checkpoint_path=checkpoints/checkpoint_00010000.ckpt \\
        +eval.n_episodes=50 +eval.mlflow_run_id=<run_id>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from playground.eval.evaluator import EvalResult, EvaluatorWithViz
from playground.policies.act_wrapper import ACTWrapper
from playground.policies.diffusion_wrapper import DiffusionWrapper

PolicyWrapper = Union[ACTWrapper, DiffusionWrapper]


def _build_policy(policy_cfg: DictConfig, device: str) -> PolicyWrapper:
    """Instantiate the appropriate policy wrapper from the Hydra policy config."""
    name: str = policy_cfg.name
    if name == "act":
        return ACTWrapper(policy_cfg, device=device)
    if name == "diffusion":
        return DiffusionWrapper(policy_cfg, device=device)
    raise ValueError(f"Unknown policy name: '{name}'. Expected 'act' or 'diffusion'.")


def _run_eval(cfg: DictConfig) -> EvalResult:
    """Core evaluation logic — callable without Hydra for testing.

    Args:
        cfg: Hydra config with ``eval.checkpoint_path``, ``eval.n_episodes``,
             ``policy.name``, ``training.device``, and ``env.name``.

    Returns:
        EvalResult written to ``eval_report.json`` in the working directory.
    """
    checkpoint_path = Path(cfg.eval.checkpoint_path)
    n_episodes: int = int(cfg.eval.get("n_episodes", 50))
    mlflow_run_id: str | None = cfg.eval.get("mlflow_run_id")
    env_name: str = cfg.env.name

    logger.info(
        f"Starting eval | policy={cfg.policy.name} "
        f"checkpoint={checkpoint_path.name} n_episodes={n_episodes}"
    )

    policy = _build_policy(cfg.policy, device=cfg.training.device)
    policy.load_checkpoint(checkpoint_path)

    visualize: bool = bool(cfg.eval.get("visualize", False))
    evaluator = EvaluatorWithViz(
        policy=policy,
        env_name=env_name,
        n_episodes=n_episodes,
        robot_name=cfg.env.name,
        policy_type=cfg.policy.name,
        mlflow_run_id=mlflow_run_id,
        visualize=visualize,
    )
    result = evaluator.evaluate()

    report_path = Path("eval_report.json")
    report_path.write_text(
        json.dumps(
            {
                "n_episodes": result.n_episodes,
                "success_rate": result.success_rate,
                "mean_reward": result.mean_reward,
                "per_episode_rewards": result.per_episode_rewards,
            },
            indent=2,
        )
    )
    logger.info(
        f"Report written to {report_path} | "
        f"success_rate={result.success_rate:.2%} mean_reward={result.mean_reward:.3f}"
    )
    return result


@hydra.main(config_path="configs", config_name="training/default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main evaluation entry point.

    Args:
        cfg: Composed Hydra config (env + policy + training + eval overrides).
    """
    logger.debug(f"Full config:\n{OmegaConf.to_yaml(cfg)}")
    _run_eval(cfg)


if __name__ == "__main__":
    main()
