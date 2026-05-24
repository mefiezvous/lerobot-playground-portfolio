# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``train._build_dataloader`` filter/sampler wiring."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf


@pytest.mark.unit
def test_build_dataloader_no_filter_no_sampler_keeps_legacy_behaviour() -> None:
    """Configs without filter/sampler keys must work exactly as before."""
    import train as train_module

    dataset_cfg = OmegaConf.create({"repo_id": "test/ds", "root": None})
    policy_cfg = OmegaConf.create({"n_obs_steps": 1, "chunk_size": 10})

    raw_ds = MagicMock(name="raw_dataset")
    dl_instance = MagicMock(name="dataloader")

    with (
        patch("train._LEROBOT_AVAILABLE", True),
        patch("train.LeRobotDataset", return_value=raw_ds) as mock_ds_cls,
        patch("train.DataLoader", return_value=dl_instance) as mock_dl,
        patch("train.instantiate") as mock_instantiate,
    ):
        result = train_module._build_dataloader(
            dataset_cfg, policy_cfg, batch_size=8, fps=20
        )

    mock_ds_cls.assert_called_once()
    mock_instantiate.assert_not_called()
    # Legacy contract: DataLoader receives dataset positionally + shuffle=True.
    args, kwargs = mock_dl.call_args
    assert args[0] is raw_ds
    assert kwargs["shuffle"] is True
    assert "sampler" not in kwargs
    assert kwargs["drop_last"] is True
    assert result is dl_instance


@pytest.mark.unit
def test_build_dataloader_filter_replaces_dataset_with_subset() -> None:
    """A configured filter must be instantiated and applied to the dataset."""
    import train as train_module

    dataset_cfg = OmegaConf.create(
        {
            "repo_id": "test/ds",
            "root": None,
            "filter": {"_target_": "mlcore.data.filter.SuccessOnlyFilter"},
        }
    )
    policy_cfg = OmegaConf.create({"n_obs_steps": 1, "chunk_size": 10})

    raw_ds = MagicMock(name="raw_dataset")
    filtered_ds = MagicMock(name="filtered_subset")
    filter_fn = MagicMock(name="filter_fn", return_value=filtered_ds)
    dl_instance = MagicMock(name="dataloader")

    def _fake_instantiate(cfg: Any, **kw: Any) -> Any:
        return filter_fn

    with (
        patch("train._LEROBOT_AVAILABLE", True),
        patch("train.LeRobotDataset", return_value=raw_ds),
        patch("train.DataLoader", return_value=dl_instance) as mock_dl,
        patch("train.instantiate", side_effect=_fake_instantiate) as mock_instantiate,
    ):
        train_module._build_dataloader(dataset_cfg, policy_cfg, batch_size=8, fps=20)

    # Filter was instantiated and called with the raw dataset.
    assert mock_instantiate.call_count == 1
    filter_fn.assert_called_once_with(raw_ds)
    # DataLoader received the filtered subset, not the raw dataset.
    args, kwargs = mock_dl.call_args
    assert args[0] is filtered_ds
    assert kwargs["shuffle"] is True


@pytest.mark.unit
def test_build_dataloader_sampler_disables_shuffle_and_is_passed_through() -> None:
    """A configured sampler is instantiated with the dataset and passed via sampler=."""
    import train as train_module

    dataset_cfg = OmegaConf.create(
        {
            "repo_id": "test/ds",
            "root": None,
            "sampler": {
                "_target_": "mlcore.data.sampler.MultiTaskBalancedSampler",
                "tasks_per_batch": 2,
                "batch_size": 8,
            },
        }
    )
    policy_cfg = OmegaConf.create({"n_obs_steps": 1, "chunk_size": 10})

    raw_ds = MagicMock(name="raw_dataset")
    sampler_instance = MagicMock(name="sampler_instance")
    dl_instance = MagicMock(name="dataloader")

    captured_kwargs: dict[str, Any] = {}

    def _fake_instantiate(cfg: Any, **kw: Any) -> Any:
        captured_kwargs.update(kw)
        return sampler_instance

    with (
        patch("train._LEROBOT_AVAILABLE", True),
        patch("train.LeRobotDataset", return_value=raw_ds),
        patch("train.DataLoader", return_value=dl_instance) as mock_dl,
        patch("train.instantiate", side_effect=_fake_instantiate),
    ):
        train_module._build_dataloader(dataset_cfg, policy_cfg, batch_size=8, fps=20)

    # instantiate was called with dataset=raw_ds (no filter) and _convert_=object.
    assert captured_kwargs.get("dataset") is raw_ds
    assert captured_kwargs.get("_convert_") == "object"

    # DataLoader received sampler=..., no shuffle kwarg.
    args, kwargs = mock_dl.call_args
    assert args[0] is raw_ds
    assert kwargs["sampler"] is sampler_instance
    assert "shuffle" not in kwargs
    assert kwargs["drop_last"] is True


@pytest.mark.unit
def test_build_dataloader_filter_then_sampler_compose() -> None:
    """Filter result must be fed into the sampler as ``dataset=``."""
    import train as train_module

    dataset_cfg = OmegaConf.create(
        {
            "repo_id": "test/ds",
            "root": None,
            "filter": {"_target_": "mlcore.data.filter.SuccessOnlyFilter"},
            "sampler": {
                "_target_": "mlcore.data.sampler.MultiTaskBalancedSampler",
                "tasks_per_batch": 2,
                "batch_size": 8,
            },
        }
    )
    policy_cfg = OmegaConf.create({"n_obs_steps": 1, "chunk_size": 10})

    raw_ds = MagicMock(name="raw_dataset")
    filtered_ds = MagicMock(name="filtered_subset")
    filter_fn = MagicMock(name="filter_fn", return_value=filtered_ds)
    sampler_instance = MagicMock(name="sampler_instance")
    dl_instance = MagicMock(name="dataloader")

    captured_sampler_kwargs: dict[str, Any] = {}
    call_order: list[str] = []

    def _fake_instantiate(cfg: Any, **kw: Any) -> Any:
        target = cfg.get("_target_") if hasattr(cfg, "get") else cfg["_target_"]
        if "filter" in target.lower():
            call_order.append("filter")
            return filter_fn
        call_order.append("sampler")
        captured_sampler_kwargs.update(kw)
        return sampler_instance

    with (
        patch("train._LEROBOT_AVAILABLE", True),
        patch("train.LeRobotDataset", return_value=raw_ds),
        patch("train.DataLoader", return_value=dl_instance) as mock_dl,
        patch("train.instantiate", side_effect=_fake_instantiate),
    ):
        train_module._build_dataloader(dataset_cfg, policy_cfg, batch_size=8, fps=20)

    assert call_order == ["filter", "sampler"]
    # Filter saw raw, sampler saw the filtered subset.
    filter_fn.assert_called_once_with(raw_ds)
    assert captured_sampler_kwargs.get("dataset") is filtered_ds
    # DataLoader received the filtered dataset + sampler.
    args, kwargs = mock_dl.call_args
    assert args[0] is filtered_ds
    assert kwargs["sampler"] is sampler_instance


@pytest.mark.unit
def test_multitask_dataset_config_is_well_formed() -> None:
    """The shipped multitask.yaml must declare filter + sampler with proper targets."""
    from pathlib import Path

    cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent
        / "configs"
        / "dataset"
        / "multitask.yaml"
    )
    assert cfg.filter._target_ == "mlcore.data.filter.SuccessOnlyFilter"
    assert cfg.sampler._target_ == "mlcore.data.sampler.MultiTaskBalancedSampler"
    assert cfg.sampler.tasks_per_batch == 4
    assert cfg.sampler.batch_size == 64
