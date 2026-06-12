# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Data-driven EnvAdapter registration from ``robot_specs/*.yaml``.

For each spec entry whose ``adapter.type`` is ``mujoco_playground``, registers
a zero-arg :class:`MujocoPlaygroundAdapter` factory under the spec's ``id`` in
:class:`robotics_platform.envs.registry.EnvAdapterRegistry`.

Entries with no ``adapter`` block, or with an unsupported ``adapter.type``,
are skipped with a warning — hardware/real-robot adapters remain code-only
(registered via :mod:`playground.envs.registrations`).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from robotics_platform.envs.registry import register_factory

from playground.envs.mujoco_playground_adapter import MujocoPlaygroundAdapter

_SUPPORTED_ADAPTER_TYPES = {"mujoco_playground"}


def register_from_yaml(specs_dir: Path) -> None:
    """Register an EnvAdapter factory for every YAML spec with a supported adapter.

    Args:
        specs_dir: Directory containing one YAML file per robot spec. If it
            does not exist, this is a no-op.
    """
    if not specs_dir.is_dir():
        return

    for path in sorted(specs_dir.glob("*.yaml")):
        entry = _load_entry(path)
        spec_id = entry["id"]
        adapter = entry.get("adapter")
        if not adapter:
            logger.debug(f"Robot spec '{spec_id}' has no adapter block — skipping registration")
            continue

        adapter_type = adapter.get("type")
        if adapter_type not in _SUPPORTED_ADAPTER_TYPES:
            logger.warning(
                f"Robot spec '{spec_id}' has unsupported adapter.type {adapter_type!r} — "
                "skipping data-driven registration (hardware adapters remain code-only)"
            )
            continue

        env_name = adapter["env_name"]
        task_description = entry.get("task", {}).get("task_description", "")
        register_factory(
            spec_id,
            functools.partial(
                MujocoPlaygroundAdapter,
                env_name=env_name,
                task_description=task_description,
            ),
        )
        logger.debug(f"Registered factory '{spec_id}' -> MujocoPlaygroundAdapter({env_name!r})")


def _load_entry(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Robot spec file {path} must contain a YAML mapping")
    return data
