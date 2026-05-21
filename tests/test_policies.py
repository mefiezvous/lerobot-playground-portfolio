# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for playground policy re-exports. Implementation tests live in ml-core."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_act_wrapper_importable() -> None:
    from playground.policies.act_wrapper import ACTWrapper

    assert ACTWrapper is not None


@pytest.mark.unit
def test_diffusion_wrapper_importable() -> None:
    from playground.policies.diffusion_wrapper import DiffusionWrapper

    assert DiffusionWrapper is not None


@pytest.mark.unit
def test_act_wrapper_is_mlcore_class() -> None:
    from mlcore.policies.act_wrapper import ACTWrapper as MlACT
    from playground.policies.act_wrapper import ACTWrapper as PubACT

    assert PubACT is MlACT


@pytest.mark.unit
def test_diffusion_wrapper_is_mlcore_class() -> None:
    from mlcore.policies.diffusion_wrapper import DiffusionWrapper as MlDiff
    from playground.policies.diffusion_wrapper import DiffusionWrapper as PubDiff

    assert PubDiff is MlDiff
