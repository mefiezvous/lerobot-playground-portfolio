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
from hydra.utils import get_class, instantiate
from loguru import logger
from mlcore.robots import get as get_robot_spec
from mlcore.robots import validate_spec_against_env
from omegaconf import DictConfig, OmegaConf
from robotics_platform.envs.registry import EnvAdapterRegistry
from torch.utils.data import DataLoader

import playground.envs.registrations  # noqa: F401 — populates EnvAdapterRegistry

try:
    from lerobot.common.datasets.lerobot_dataset import (  # type: ignore[import-untyped]
        LeRobotDataset,
    )

    _LEROBOT_AVAILABLE = True
except ImportError:
    LeRobotDataset = None  # type: ignore[assignment,misc]
    _LEROBOT_AVAILABLE = False

from playground.training.trainer import PolicyWrapper, Trainer


def _build_policy(policy_cfg: DictConfig, device: str) -> PolicyWrapper:
    """Instantiate the appropriate policy wrapper via Hydra ``_target_`` resolution.

    The ``_target_`` field on the policy sub-config selects the wrapper class
    (e.g. ``playground.policies.act_wrapper.ACTWrapper``). The wrapper receives
    the full ``policy_cfg`` as its ``cfg`` argument so it can read fields such
    as ``n_obs_steps``, ``input_shapes``, ``pretrained_repo``, etc.

    Args:
        policy_cfg: The ``cfg.policy`` sub-config (must contain ``_target_``).
        device: PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).

    Raises:
        ValueError: If ``_target_`` is missing or cannot be resolved.
    """
    target: str | None = policy_cfg.get("_target_")
    if not target:
        raise ValueError(
            "Policy config is missing the '_target_' field. "
            "Add e.g. '_target_: playground.policies.act_wrapper.ACTWrapper' "
            "to your configs/policy/<name>.yaml under the 'policy:' block."
        )
    try:
        wrapper_cls = get_class(target)
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError) as exc:
        raise ValueError(
            f"Failed to resolve policy '_target_' = '{target}'. "
            f"Ensure the dotted path points to an importable class. ({exc})"
        ) from exc
    return wrapper_cls(policy_cfg, device=device)  # type: ignore[no-any-return]


def _build_dataloader(
    dataset_cfg: DictConfig,
    policy_cfg: DictConfig,
    batch_size: int,
    fps: int,
) -> DataLoader[Any]:
    """Load a LeRobotDataset and wrap it in a DataLoader with temporal chunking.

    Optionally applies a Hydra-instantiated ``filter`` (e.g.
    :class:`mlcore.data.filter.SuccessOnlyFilter`) to the dataset and/or a
    Hydra-instantiated ``sampler`` (e.g.
    :class:`mlcore.data.sampler.MultiTaskBalancedSampler`) to the
    DataLoader. Both keys are optional — configs that omit them keep the
    previous behaviour (full dataset, shuffled DataLoader).

    Args:
        dataset_cfg: Dataset config (``repo_id``, ``root``, optional
            ``filter``, optional ``sampler``).
        policy_cfg: Policy config providing ``n_obs_steps`` and
            ``chunk_size`` / ``n_action_steps``.
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
    dataset: Any = LeRobotDataset(
        dataset_cfg.repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
    )

    filter_cfg: Any = dataset_cfg.get("filter")
    if filter_cfg is not None:
        filter_fn: Any = instantiate(filter_cfg)
        logger.info("Applying dataset filter: {}", type(filter_fn).__name__)
        dataset = filter_fn(dataset)

    sampler_cfg: Any = dataset_cfg.get("sampler")
    if sampler_cfg is not None:
        sampler: Any = instantiate(sampler_cfg, dataset=dataset, _convert_="object")
        logger.info("Using custom sampler: {}", type(sampler).__name__)
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, drop_last=True
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

    # Validate RobotSpec ↔ EnvAdapter shape before doing any heavy work.
    # Note: instantiating the env (MuJoCo Playground sim) just for shape
    # inspection is non-trivial, but it only runs once at startup — the cost
    # is paid to guarantee fail-fast on misconfigured robots.
    logger.info("Validating robot spec ↔ env for '{}'...", cfg.env.name)
    _env_cls = EnvAdapterRegistry.get(cfg.env.name)
    _env = _env_cls()
    try:
        _spec = get_robot_spec(cfg.env.name)
        validate_spec_against_env(_spec, _env)
    finally:
        _env.close()
    logger.info("Spec validation OK")

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
