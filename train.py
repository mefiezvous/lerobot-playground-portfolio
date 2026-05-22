# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Training entrypoint — Hydra-configurable CLI.

Usage::

    # ACT policy, default config
    python train.py logging.run_name=act_run_001

    # Diffusion policy override
    python train.py --config-name training/default policy=diffusion \\
        logging.run_name=diff_run_001

    # Kaggle profile (2×T4, 100 k steps, checkpoint resume)
    python train.py --config-name training/kaggle policy=act \\
        logging.run_name=kaggle_act_001

    # Resume from checkpoint (auto-detected from checkpoint_dir)
    python train.py logging.run_name=act_run_001 training.checkpoint_dir=checkpoints/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

try:
    from lerobot.common.datasets.lerobot_dataset import (  # type: ignore[import-untyped]
        LeRobotDataset,
    )

    _LEROBOT_AVAILABLE = True
except ImportError:
    LeRobotDataset = None  # type: ignore[assignment,misc]
    _LEROBOT_AVAILABLE = False

from playground.policies.act_wrapper import ACTWrapper
from playground.policies.diffusion_wrapper import DiffusionWrapper
from playground.training.trainer import PolicyWrapper, Trainer


def _build_policy(policy_cfg: DictConfig, device: str) -> PolicyWrapper:
    """Instantiate the appropriate policy wrapper from the Hydra policy config."""
    name: str = policy_cfg.name
    if name == "act":
        return ACTWrapper(policy_cfg, device=device)
    if name == "diffusion":
        return DiffusionWrapper(policy_cfg, device=device)
    raise ValueError(f"Unknown policy name: '{name}'. Expected 'act' or 'diffusion'.")


def _build_dataloader(
    dataset_cfg: DictConfig,
    policy_cfg: DictConfig,
    batch_size: int,
    fps: int,
) -> DataLoader[Any]:
    """Load a LeRobotDataset and wrap it in a DataLoader with temporal chunking.

    Args:
        dataset_cfg: Dataset config (repo_id, root).
        policy_cfg: Policy config providing n_obs_steps and chunk_size/n_action_steps.
        batch_size: Training batch size.
        fps: Environment control frequency (Hz) for timestamp computation.
    """
    if not _LEROBOT_AVAILABLE:
        raise ImportError("lerobot is required. Install with: uv sync")

    n_obs_steps: int = policy_cfg.get("n_obs_steps", 1)
    chunk_size: int = int(policy_cfg.get("chunk_size", policy_cfg.get("n_action_steps", 1)))

    delta_timestamps: dict[str, list[float]] = {
        "observation.state": [(i - n_obs_steps + 1) / fps for i in range(n_obs_steps)],
        "action": [i / fps for i in range(chunk_size)],
    }

    root: Path | None = Path(dataset_cfg.root) if dataset_cfg.get("root") else None
    dataset = LeRobotDataset(
        dataset_cfg.repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


@hydra.main(config_path="configs", config_name="training/default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training entry point.

    Args:
        cfg: Composed Hydra config (env + policy + training + logging).
    """
    logger.info(
        f"Starting training | policy={cfg.policy.name} "
        f"steps={cfg.training.total_steps} device={cfg.training.device}"
    )
    logger.debug(f"Full config:\n{OmegaConf.to_yaml(cfg)}")

    policy = _build_policy(cfg.policy, device=cfg.training.device)
    dataloader = _build_dataloader(
        cfg.dataset,
        cfg.policy,
        batch_size=cfg.training.batch_size,
        fps=cfg.env.fps,
    )
    trainer = Trainer(
        cfg,
        policy,
        dataloader,
        robot_name=cfg.env.name,
        policy_type=cfg.policy.name,
        hf_repo_id=cfg.get("hf_repo_id"),
    )
    trainer.train()


if __name__ == "__main__":
    main()
