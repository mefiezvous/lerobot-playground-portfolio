# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for playground trainer re-exports. Implementation tests live in ml-core."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_trainer_importable() -> None:
    from playground.training.trainer import Trainer

    assert Trainer is not None


@pytest.mark.unit
def test_policy_wrapper_importable() -> None:
    from playground.training.trainer import PolicyWrapper

    assert PolicyWrapper is not None


@pytest.mark.unit
def test_trainer_is_mlcore_class() -> None:
    from mlcore.training.trainer import Trainer as MlTrainer
    from playground.training.trainer import Trainer as PubTrainer

    assert PubTrainer is MlTrainer


@pytest.mark.unit
def test_policy_wrapper_is_mlcore_protocol() -> None:
    from mlcore.training.trainer import PolicyWrapper as MlPW
    from playground.training.trainer import PolicyWrapper as PubPW

    assert PubPW is MlPW
