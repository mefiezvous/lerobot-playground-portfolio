# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``playground.envs.yaml_registrations.register_from_yaml``."""

from __future__ import annotations

from pathlib import Path

import pytest
from robotics_platform.envs.registry import EnvAdapterRegistry

from playground.envs.mujoco_playground_adapter import MujocoPlaygroundAdapter
from playground.envs.yaml_registrations import register_from_yaml

_MUJOCO_PLAYGROUND_SPEC = """
id: yaml_reg_mp
spec:
  n_joints: 6
  obs_keys: [ee_pos, target_pos]
  action_dim: 6
  target_pos_key: target_pos
task:
  task_description: "Do the thing"
adapter:
  type: mujoco_playground
  env_name: SomeEnv
"""

_NO_ADAPTER_SPEC = """
id: yaml_reg_no_adapter
spec:
  n_joints: 6
  obs_keys: [ee_pos, target_pos]
  action_dim: 6
  target_pos_key: target_pos
adapter: null
"""

_UNSUPPORTED_ADAPTER_SPEC = """
id: yaml_reg_unsupported
spec:
  n_joints: 6
  obs_keys: [ee_pos, target_pos]
  action_dim: 6
  target_pos_key: target_pos
adapter:
  type: real_hardware
  env_name: SomeEnv
"""


@pytest.mark.unit
def test_register_from_yaml_noop_for_missing_dir(tmp_path: Path) -> None:
    register_from_yaml(tmp_path / "does_not_exist")
    assert "yaml_reg_mp" not in EnvAdapterRegistry.list_adapters()


@pytest.mark.unit
def test_register_from_yaml_registers_mujoco_playground_factory(tmp_path: Path) -> None:
    (tmp_path / "spec.yaml").write_text(_MUJOCO_PLAYGROUND_SPEC, encoding="utf-8")

    register_from_yaml(tmp_path)

    factory = EnvAdapterRegistry.get("yaml_reg_mp")
    assert factory.func is MujocoPlaygroundAdapter
    assert factory.keywords == {"env_name": "SomeEnv", "task_description": "Do the thing"}


@pytest.mark.unit
def test_register_from_yaml_skips_entry_without_adapter(tmp_path: Path) -> None:
    (tmp_path / "spec.yaml").write_text(_NO_ADAPTER_SPEC, encoding="utf-8")

    register_from_yaml(tmp_path)

    assert "yaml_reg_no_adapter" not in EnvAdapterRegistry.list_adapters()


@pytest.mark.unit
def test_register_from_yaml_skips_unsupported_adapter_type(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from loguru import logger

    (tmp_path / "spec.yaml").write_text(_UNSUPPORTED_ADAPTER_SPEC, encoding="utf-8")

    handler_id = logger.add(caplog.handler, format="{message}", level="WARNING")
    try:
        with caplog.at_level("WARNING"):
            register_from_yaml(tmp_path)
    finally:
        logger.remove(handler_id)

    assert "yaml_reg_unsupported" not in EnvAdapterRegistry.list_adapters()
    assert any("unsupported adapter.type" in rec.message for rec in caplog.records)
