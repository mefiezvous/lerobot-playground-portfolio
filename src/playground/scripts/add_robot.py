# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""CLI to scaffold a new robot: generates robot config YAML and env stub.

Usage::

    python add_robot.py <name> --n-joints N --action-dim D \\
        [--ee-pos-key K] [--target-pos-key K]

Example::

    python add_robot.py cube_reach_v2 --n-joints 7 --action-dim 7
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import yaml
from loguru import logger

from mlcore.robots import RobotSpec, build_train_config

# Paths relative to this script: src/playground/scripts/ → src/playground/
_PLAYGROUND_DIR = Path(__file__).parent.parent
_CONFIGS_DIR = _PLAYGROUND_DIR / "configs" / "robot"
_ENVS_DIR = _PLAYGROUND_DIR / "envs"

_USAGE = (
    "Usage: python add_robot.py <name> --n-joints N --action-dim D "
    "[--ee-pos-key K] [--target-pos-key K]"
)


def _parse_argv() -> dict[str, str]:
    """Parse sys.argv[1:] as positional name + --key value pairs."""
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        logger.error(f"Missing robot name.\n{_USAGE}")
        sys.exit(1)

    parsed: dict[str, str] = {"name": args[0]}
    i = 1
    while i < len(args):
        if not args[i].startswith("--"):
            logger.error(f"Expected --flag, got {args[i]!r}.\n{_USAGE}")
            sys.exit(1)
        if i + 1 >= len(args):
            logger.error(f"Flag {args[i]} has no value.\n{_USAGE}")
            sys.exit(1)
        key = args[i].lstrip("-").replace("-", "_")
        parsed[key] = args[i + 1]
        i += 2
    return parsed


def _build_spec(params: dict[str, str]) -> RobotSpec:
    for required in ("n_joints", "action_dim"):
        if required not in params:
            logger.error(f"Missing required flag --{required.replace('_', '-')}.\n{_USAGE}")
            sys.exit(1)

    ee_pos_key = params.get("ee_pos_key", "ee_pos")
    target_pos_key = params.get("target_pos_key", "cube_pos")

    return RobotSpec(
        name=params["name"],
        n_joints=int(params["n_joints"]),
        obs_keys=[ee_pos_key, target_pos_key],
        action_dim=int(params["action_dim"]),
        ee_pos_key=ee_pos_key,
        target_pos_key=target_pos_key,
    )


def _write_config(spec: RobotSpec) -> Path:
    _CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_train_config(spec, policy_type="act")
    out = _CONFIGS_DIR / f"{spec.name}.yaml"
    out.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return out


def _write_env_stub(spec: RobotSpec) -> Path:
    _ENVS_DIR.mkdir(parents=True, exist_ok=True)
    class_name = "".join(part.capitalize() for part in spec.name.split("_"))
    stub = textwrap.dedent(f"""\
        # SPDX-FileCopyrightText: 2026 Arthur Mouraud
        # SPDX-License-Identifier: Apache-2.0
        \"\"\"Stub environment for {spec.name}.

        TODO:
          1. Choose a MuJoCo Playground base env (e.g. PandaPickCube).
          2. Subclass it and override compute_reward().
          3. Register: mp.register("{class_name}", {class_name})
        \"\"\"

        from __future__ import annotations

        # from mujoco_playground._src.manipulation.panda_pick_cube import PandaPickCube
        # from mujoco_playground import register
        # from typing import Any
        # import jax
        #
        # class {class_name}(PandaPickCube):
        #     def compute_reward(self, obs: dict[str, Any], info: dict[str, Any]) -> jax.Array:
        #         raise NotImplementedError
        #
        # register("{class_name}", {class_name})
    """)
    out = _ENVS_DIR / f"{spec.name}.py"
    out.write_text(stub, encoding="utf-8")
    return out


def main() -> None:
    params = _parse_argv()
    spec = _build_spec(params)

    config_path = _write_config(spec)
    env_path = _write_env_stub(spec)

    logger.info(f"Created config  : {config_path}")
    logger.info(f"Created env stub: {env_path}")
    logger.info(
        f"\nNext steps for '{spec.name}':\n"
        f"  1. Edit {env_path.name} — subclass a base env and implement compute_reward()\n"
        f"  2. Edit {config_path.name} — adjust obs_keys, success_threshold, etc.\n"
        f"  3. Add the spec to ml-core/src/mlcore/robots/specs/{spec.name}.py\n"
        f"  4. Run: python train.py robot={spec.name}"
    )


if __name__ == "__main__":
    main()
