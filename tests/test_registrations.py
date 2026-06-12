# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Guardian tests for EnvAdapter nomenclature conventions.

Enforces the rules documented in
``robotics-platform-template/docs/GLOSSARY.md``:

* Registry keys are snake_case (``cube_reach_v1``).
* For hand-written ``@register()``'d adapter classes, class names are
  PascalCase with the ``Adapter`` suffix, and the PascalCase of the
  registry key matches the class name (minus the ``Adapter`` suffix).

Data-driven ``robot_specs/*.yaml`` entries register a ``functools.partial``
factory around :class:`MujocoPlaygroundAdapter` rather than a dedicated
class, so the class-naming rule does not apply to them.

This prevents nomenclature drift when adding new envs.
"""

from __future__ import annotations

import functools
import re

import pytest
from robotics_platform.envs.registry import EnvAdapterRegistry

# Importing the module has the side-effect of populating the registry.
import playground.envs.registrations  # noqa: F401
from playground.envs.mujoco_playground_adapter import MujocoPlaygroundAdapter

SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*[a-z0-9]$")


def _snake_to_pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


@pytest.mark.unit
class TestEnvAdapterNomenclature:
    """Each registered env must follow the snake_case ↔ PascalCase rules."""

    def test_registry_is_non_empty(self) -> None:
        assert EnvAdapterRegistry.list_adapters(), (
            "EnvAdapterRegistry is empty — playground.envs.registrations not imported?"
        )

    @pytest.mark.parametrize("name", EnvAdapterRegistry.list_adapters())
    def test_registry_key_is_snake_case(self, name: str) -> None:
        assert SNAKE_CASE.match(name), (
            f"Registry key '{name}' is not snake_case. "
            "Expected lowercase letters, digits and underscores (see docs/GLOSSARY.md)."
        )

    @pytest.mark.parametrize("name", EnvAdapterRegistry.list_adapters())
    def test_class_name_matches_key(self, name: str) -> None:
        adapter = EnvAdapterRegistry.get(name)
        if not isinstance(adapter, type):
            # Data-driven (robot_specs/*.yaml) registrations map to a
            # functools.partial factory around MujocoPlaygroundAdapter —
            # the snake_case ↔ PascalCase class-naming convention only
            # applies to hand-written @register()'d adapter classes.
            pytest.skip(f"'{name}' is a data-driven factory, not an adapter class")

        cls = adapter
        expected = f"{_snake_to_pascal(name)}Adapter"
        assert cls.__name__ == expected, (
            f"Registry key '{name}' maps to class '{cls.__name__}', "
            f"expected '{expected}' per snake_case → PascalCase + 'Adapter' convention "
            "(see robotics-platform-template/docs/GLOSSARY.md)."
        )

    def test_cube_reach_v1_registered_from_yaml_factory(self) -> None:
        """``cube_reach_v1`` is registered by the data-driven YAML loader.

        It used to be a hand-written ``CubeReachV1Adapter`` class (removed in
        P2 cleanup); the single source of truth is now
        ``robot_specs/cube_reach_v1.yaml``. ``get`` must return a
        ``functools.partial`` factory around :class:`MujocoPlaygroundAdapter`,
        not an adapter class.
        """
        factory = EnvAdapterRegistry.get("cube_reach_v1")
        assert not isinstance(factory, type), (
            "cube_reach_v1 should be a data-driven factory, not an adapter class"
        )
        assert isinstance(factory, functools.partial)
        assert factory.func is MujocoPlaygroundAdapter
        assert factory.keywords.get("env_name") == "CubeReachV1"
        assert factory.keywords.get("task_description") == "Reach the cube"
