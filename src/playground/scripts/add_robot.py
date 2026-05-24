# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""CLI to scaffold a new robot end-to-end.

Generates every file needed to onboard a new robot in one shot:
  1. ``src/playground/configs/robot/<name>.yaml``        — robot training config
  2. ``src/playground/envs/<name>.py``                    — env stub
  3. ``ml-core/src/mlcore/robots/specs/<name>.py``        — RobotSpec + register
  4. Appends an EnvAdapter registration in
     ``src/playground/envs/registrations.py`` (idempotent)
  5. ``configs/env/<name>.yaml``                          — Hydra env config
  6. ``configs/dataset/<name>.yaml``                      — Hydra dataset config

Usage::

    python -m playground.scripts.add_robot <name> \\
        --n-joints N --action-dim D \\
        [--obs-keys "ee_pos,target_pos,joints"] \\
        [--ee-pos-key K] [--target-pos-key K] \\
        [--task-description "..."] \\
        [--fps 20] [--episode-length 200] \\
        [--dry-run]
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import yaml
from loguru import logger
from mlcore.robots import RobotSpec, build_train_config

# Paths relative to this script: src/playground/scripts/ → src/playground/
_SCRIPT_DIR = Path(__file__).resolve().parent
_PLAYGROUND_DIR = _SCRIPT_DIR.parent
_REPO_DIR = _PLAYGROUND_DIR.parent.parent  # lerobot-playground-portfolio/
_WORKSPACE_DIR = _REPO_DIR.parent  # robotics-workspace/

_ROBOT_CFG_DIR = _PLAYGROUND_DIR / "configs" / "robot"
_ENVS_DIR = _PLAYGROUND_DIR / "envs"
_REGISTRATIONS_FILE = _ENVS_DIR / "registrations.py"

_HYDRA_ENV_DIR = _REPO_DIR / "configs" / "env"
_HYDRA_DATASET_DIR = _REPO_DIR / "configs" / "dataset"

_MLCORE_SPECS_DIR = _WORKSPACE_DIR / "ml-core" / "src" / "mlcore" / "robots" / "specs"

_USAGE = (
    "Usage: python -m playground.scripts.add_robot <name> "
    "--n-joints N --action-dim D "
    "[--obs-keys ee_pos,target_pos,joints] "
    "[--ee-pos-key K] [--target-pos-key K] "
    "[--task-description '...'] "
    "[--fps 20] [--episode-length 200] [--dry-run]"
)


def _parse_argv() -> dict[str, str]:
    """Parse sys.argv[1:] as positional name + --key value pairs and --flags."""
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        logger.error(f"Missing robot name.\n{_USAGE}")
        sys.exit(1)

    parsed: dict[str, str] = {"name": args[0]}
    bool_flags = {"dry_run"}
    i = 1
    while i < len(args):
        if not args[i].startswith("--"):
            logger.error(f"Expected --flag, got {args[i]!r}.\n{_USAGE}")
            sys.exit(1)
        key = args[i].lstrip("-").replace("-", "_")
        if key in bool_flags:
            parsed[key] = "1"
            i += 1
            continue
        if i + 1 >= len(args):
            logger.error(f"Flag {args[i]} has no value.\n{_USAGE}")
            sys.exit(1)
        parsed[key] = args[i + 1]
        i += 2
    return parsed


def _class_name(snake: str) -> str:
    return "".join(part.capitalize() for part in snake.split("_"))


def _build_spec(params: dict[str, str]) -> RobotSpec:
    for required in ("n_joints", "action_dim"):
        if required not in params:
            logger.error(f"Missing required flag --{required.replace('_', '-')}.\n{_USAGE}")
            sys.exit(1)

    ee_pos_key = params.get("ee_pos_key", "ee_pos")
    target_pos_key = params.get("target_pos_key", "cube_pos")

    if "obs_keys" in params:
        obs_keys = [k.strip() for k in params["obs_keys"].split(",") if k.strip()]
    else:
        obs_keys = [ee_pos_key, target_pos_key]

    return RobotSpec(
        name=params["name"],
        n_joints=int(params["n_joints"]),
        obs_keys=obs_keys,
        action_dim=int(params["action_dim"]),
        ee_pos_key=ee_pos_key,
        target_pos_key=target_pos_key,
    )


def _render_robot_config(spec: RobotSpec) -> str:
    cfg = build_train_config(spec, policy_type="act")
    return str(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


def _render_env_stub(spec: RobotSpec) -> str:
    class_name = _class_name(spec.name)
    return textwrap.dedent(f"""\
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


def _render_spec_file(spec: RobotSpec) -> str:
    const_name = spec.name.upper()
    obs_keys_repr = repr(list(spec.obs_keys))
    return textwrap.dedent(f"""\
        # SPDX-FileCopyrightText: 2026 Arthur Mouraud
        # SPDX-License-Identifier: Apache-2.0
        \"\"\"Spec for the {_class_name(spec.name)} task.\"\"\"

        from __future__ import annotations

        from mlcore.robots.base import RobotSpec
        from mlcore.robots.registry import register

        {const_name} = RobotSpec(
            name="{spec.name}",
            n_joints={spec.n_joints},
            obs_keys={obs_keys_repr},
            action_dim={spec.action_dim},
            ee_pos_key="{spec.ee_pos_key}",
            target_pos_key="{spec.target_pos_key}",
        )
        register({const_name})
    """)


def _render_env_yaml(spec: RobotSpec, fps: int, episode_length: int, task_description: str) -> str:
    payload = {
        "env": {
            "name": _class_name(spec.name),
            "fps": fps,
            "episode_length": episode_length,
            "task_description": task_description,
            "success_threshold": spec.success_threshold,
            "seed": 42,
        }
    }
    header = "# SPDX-FileCopyrightText: 2026 Arthur Mouraud\n"
    header += "# SPDX-License-Identifier: Apache-2.0\n"
    header += "# @package _global_\n"
    return header + str(yaml.dump(payload, default_flow_style=False, sort_keys=False))


def _render_dataset_yaml(spec: RobotSpec) -> str:
    payload = {
        "repo_id": f"mefiezvous/{spec.name}-dataset",
        "task_id": spec.name,
        "root": f"data/{spec.name}",
    }
    header = "# SPDX-FileCopyrightText: 2026 Arthur Mouraud\n"
    header += "# SPDX-License-Identifier: Apache-2.0\n"
    return header + str(yaml.dump(payload, default_flow_style=False, sort_keys=False))


def _render_registration_snippet(spec: RobotSpec, task_description: str) -> str:
    class_name = _class_name(spec.name)
    desc = task_description or f"Task: {spec.name}"
    return textwrap.dedent(f"""

        @register("mujoco_pgnd:{spec.name}")
        class {class_name}Adapter(MujocoPlaygroundAdapter):
            \"\"\"Zero-arg adapter for the {class_name} MuJoCo Playground env.\"\"\"

            def __init__(self) -> None:
                super().__init__(
                    env_name="{class_name}",
                    task_description="{desc}",
                )
    """)


def _registration_already_present(name: str) -> bool:
    if not _REGISTRATIONS_FILE.exists():
        return False
    needle = f'@register("mujoco_pgnd:{name}")'
    return needle in _REGISTRATIONS_FILE.read_text(encoding="utf-8")


def _preview(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:100]
    return "(empty)"


def _write_file(path: Path, content: str, dry_run: bool) -> str:
    if dry_run:
        logger.info(f"[dry-run] would write {path}")
        logger.info(f"          preview: {_preview(content)}")
        return "planned"
    if path.exists():
        logger.info(f"[skip] {path} already exists")
        return "skipped"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info(f"[write] {path}")
    return "written"


def _append_registration(spec: RobotSpec, task_description: str, dry_run: bool) -> str:
    snippet = _render_registration_snippet(spec, task_description)
    if dry_run:
        logger.info(f"[dry-run] would append registration to {_REGISTRATIONS_FILE}")
        logger.info(f'          preview: @register("mujoco_pgnd:{spec.name}")')
        return "planned"
    if _registration_already_present(spec.name):
        logger.info(f"[skip] registration for {spec.name} already present")
        return "skipped"
    with _REGISTRATIONS_FILE.open("a", encoding="utf-8") as f:
        f.write(snippet)
    logger.info(f"[append] {_REGISTRATIONS_FILE}")
    return "appended"


def main() -> None:
    params = _parse_argv()
    dry_run = "dry_run" in params
    fps = int(params.get("fps", "20"))
    episode_length = int(params.get("episode_length", "200"))
    task_description = params.get("task_description", "")

    spec = _build_spec(params)

    # 1. Robot config YAML (playground)
    robot_cfg_path = _ROBOT_CFG_DIR / f"{spec.name}.yaml"
    _write_file(robot_cfg_path, _render_robot_config(spec), dry_run)

    # 2. Env stub (playground)
    env_stub_path = _ENVS_DIR / f"{spec.name}.py"
    _write_file(env_stub_path, _render_env_stub(spec), dry_run)

    # 3. RobotSpec in ml-core
    spec_path = _MLCORE_SPECS_DIR / f"{spec.name}.py"
    _write_file(spec_path, _render_spec_file(spec), dry_run)

    # 4. Append EnvAdapter registration
    _append_registration(spec, task_description, dry_run)

    # 5. Hydra env config
    env_yaml_path = _HYDRA_ENV_DIR / f"{spec.name}.yaml"
    _write_file(
        env_yaml_path,
        _render_env_yaml(spec, fps, episode_length, task_description),
        dry_run,
    )

    # 6. Hydra dataset config
    dataset_yaml_path = _HYDRA_DATASET_DIR / f"{spec.name}.yaml"
    _write_file(dataset_yaml_path, _render_dataset_yaml(spec), dry_run)

    if dry_run:
        logger.info(f"\n[dry-run] no files written for '{spec.name}'.")
        return

    logger.info(
        f"\nScaffolded '{spec.name}'. Remaining manual steps:\n"
        f"  1. Implement compute_reward() in {env_stub_path.name}\n"
        f"  2. Place the MJCF/URDF asset and update the env's asset path"
    )


if __name__ == "__main__":
    main()
