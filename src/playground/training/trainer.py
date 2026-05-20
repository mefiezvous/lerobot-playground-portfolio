# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Hydra-configurable training loop with MLflow logging and checkpoint resume."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Union

import mlflow
import torch
from loguru import logger
from omegaconf import DictConfig

from playground.policies.act_wrapper import ACTWrapper
from playground.policies.diffusion_wrapper import DiffusionWrapper

PolicyWrapper = Union[ACTWrapper, DiffusionWrapper]


class Trainer:
    """Imitation-learning training loop over a LeRobotDataset.

    Handles:
    - Checkpoint resume (detects latest ``.ckpt`` in ``checkpoint_dir``)
    - MLflow metric logging at every ``save_every_steps``
    - Optional WandB logging gated by ``WANDB_API_KEY`` environment variable

    Args:
        cfg: Full Hydra config (``training.*``, ``logging.*`` keys used).
        policy: Wrapped lerobot policy with a compatible interface.
        dataloader: Iterable that yields batched tensor dicts. Cycled
            automatically when exhausted.
    """

    def __init__(
        self,
        cfg: DictConfig,
        policy: PolicyWrapper,
        dataloader: Iterable[dict[str, torch.Tensor]],
    ) -> None:
        self._cfg = cfg
        self._policy = policy
        self._dataloader = dataloader
        self._step = 0

        self._setup_mlflow()
        self._resume_from_checkpoint()
        self._setup_optimizer()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_mlflow(self) -> None:
        mlflow.set_tracking_uri(self._cfg.logging.mlflow_tracking_uri)
        mlflow.set_experiment(self._cfg.logging.experiment_name)
        mlflow.start_run(run_name=self._cfg.logging.run_name)
        logger.info(
            f"MLflow run started | experiment={self._cfg.logging.experiment_name} "
            f"run={self._cfg.logging.run_name}"
        )

    def _setup_optimizer(self) -> None:
        self._optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr=self._cfg.training.lr,
        )

    def _setup_wandb(self) -> None:
        if not os.environ.get("WANDB_API_KEY"):
            self._wandb_enabled = False
            return
        try:
            import wandb  # type: ignore[import-untyped]  # noqa: PLC0415

            wandb.init(
                project=self._cfg.logging.experiment_name,
                name=self._cfg.logging.run_name,
            )
            self._wandb: Any = wandb
            self._wandb_enabled = True
            logger.info("WandB logging enabled")
        except ImportError:
            logger.warning("WANDB_API_KEY is set but wandb is not installed — skipping")
            self._wandb_enabled = False

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _detect_latest_checkpoint(self) -> Path | None:
        """Return the lexicographically last ``.ckpt`` file, or ``None``."""
        ckpt_dir = Path(self._cfg.training.checkpoint_dir)
        if not ckpt_dir.exists():
            return None
        ckpts = sorted(ckpt_dir.glob("checkpoint_*.ckpt"))
        return ckpts[-1] if ckpts else None

    def _resume_from_checkpoint(self) -> None:
        ckpt = self._detect_latest_checkpoint()
        if ckpt is None:
            return
        self._policy.load_checkpoint(ckpt)
        # Filename format: checkpoint_XXXXXXXX.ckpt → extract step
        self._step = int(ckpt.stem.split("_")[-1])
        logger.info(f"Resumed from {ckpt.name} at step {self._step}")

    def _save_checkpoint(self, step: int) -> None:
        ckpt_dir = Path(self._cfg.training.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"checkpoint_{step:08d}.ckpt"
        self._policy.save(ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path.name}")

    # ------------------------------------------------------------------
    # Metric logging
    # ------------------------------------------------------------------

    def _log_metrics(
        self,
        step: int,
        loss: float,
        reward: float = 0.0,
        success_rate: float = 0.0,
    ) -> None:
        metrics = {"loss": loss, "reward": reward, "success_rate": success_rate}
        mlflow.log_metrics(metrics, step=step)
        if getattr(self, "_wandb_enabled", False):
            self._wandb.log(metrics, step=step)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the training loop from the current step to ``total_steps``."""
        self._setup_wandb()
        total = self._cfg.training.total_steps
        save_every = self._cfg.training.save_every_steps

        dataloader_iter: Iterator[dict[str, torch.Tensor]] = iter(self._dataloader)

        try:
            for step in range(self._step, total):
                # Cycle dataloader when exhausted (dataset smaller than total_steps)
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(self._dataloader)
                    batch = next(dataloader_iter)

                self._optimizer.zero_grad()
                loss_tensor = self._policy.forward(batch)
                loss_tensor.backward()
                self._optimizer.step()

                current_step = step + 1
                if current_step % save_every == 0:
                    loss_val = float(loss_tensor.detach())
                    self._save_checkpoint(current_step)
                    self._log_metrics(current_step, loss_val)
                    logger.info(
                        f"Step {current_step}/{total} | loss={loss_val:.4f}"
                    )
        finally:
            mlflow.end_run()

        logger.info("Training complete.")
