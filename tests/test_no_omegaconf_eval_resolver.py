# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Regression guard: no OmegaConf eval-style resolver registered at import time.

LRB-007 — The OmegaConf RCE surface (custom resolvers, ${oc.env:VAR},
${oc.create:...}, eval-style resolvers) must not be exposed.  This test
asserts that importing the top-level playground package does NOT call
OmegaConf.register_new_resolver, preventing accidental introduction of
eval-accessible interpolations in the future.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_no_register_new_resolver_at_import() -> None:
    """Importing playground must not call OmegaConf.register_new_resolver."""
    mock_register = MagicMock()

    with patch("omegaconf.OmegaConf.register_new_resolver", mock_register):
        # Force a fresh import to trigger any module-level side-effects.
        import playground  # noqa: F401

        importlib.reload(playground)

    (
        mock_register.assert_not_called(),
        (
            "OmegaConf.register_new_resolver was called at import time — "
            "this exposes an eval-style interpolation surface (LRB-007)."
        ),
    )
