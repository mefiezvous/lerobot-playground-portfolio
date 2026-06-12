<!--
SPDX-FileCopyrightText: 2026 Arthur Mouraud
SPDX-License-Identifier: Apache-2.0
-->

# ADR-001 — `robot_specs/*.yaml` data-driven registry for MuJoCo Playground envs

- **Status**: Implemented 2026-06-10
- **Deciders**: Arthur Mouraud
- **Scope**: `lerobot-playground-portfolio` (`src/playground/envs/`, new `robot_specs/`), consumes `ml-core/docs/adr/ADR-001-yaml-spec-loader.md` and `robotics-platform-template/docs/adr/ADR-002-env-adapter-factory-registration.md`. Consumed by `orchestrator` (P1, `/api/v1/robots`).

## Context

The orchestrator wants to let users declare, branch and edit robots from the frontend (`id`/`name`/`parent_id` lineage model — see workspace memory `arch_orchestrator.md`). Its `api` service mounts sibling repos read-only and never writes Python source. The existing `add_robot.py` flow generates a `RobotSpec` Python module, an `EnvAdapter` subclass in `registrations.py`, and Hydra `configs/env/*.yaml` + `configs/dataset/*.yaml` — none of which the orchestrator can produce safely.

`ml-core` (ADR-001) now offers `load_specs_from_dir()` to build `RobotSpec` instances from YAML, and `robotics-platform-template` (ADR-002) now offers `register_factory()` to register `EnvAdapter` factories without a hand-written class. This repo wires both together for its MuJoCo Playground envs.

## Decision

Introduce `robot_specs/` at the repo root (sibling of `configs/`, outside `src/`) holding one YAML file per robot spec version, each consolidating what was previously spread across a `mlcore.robots.specs.*` module, `configs/env/*.yaml`, `configs/dataset/*.yaml`, and a `registrations.py` adapter class:

```yaml
id: cube_reach_v1          # technical key == EnvAdapterRegistry / mlcore registry key
name: cube_reach            # lineage label (UI grouping)
parent_id: null             # branch lineage — null for roots
spec: {...}                 # mirrors RobotSpec fields
task: {task_description, fps, episode_length, seed, ...}   # "objectifs"
adapter: {type: mujoco_playground, env_name: CubeReachV1}   # null/absent = no auto-registration
dataset: {repo_id, task_id, root}
```

New module `src/playground/envs/yaml_registrations.py::register_from_yaml(specs_dir)`: for every entry with `adapter.type == "mujoco_playground"`, calls `register_factory(entry["id"], functools.partial(MujocoPlaygroundAdapter, env_name=..., task_description=...))`. Entries with no `adapter` block are skipped silently (debug log) — this is the expected shape for lineage stubs (e.g. `cube_reach_v2`) that don't yet have a runnable env. Entries with an `adapter.type` other than `mujoco_playground` are skipped with a **warning** — hardware/real-robot adapters are explicitly out of scope for data-driven registration and remain code-only (`registrations.py`, `_private/my-robot-stack`).

`src/playground/envs/registrations.py` — already imported for its side effects by `playground.data.pipeline` and `collect.py`/`train.py` — now also calls, after its existing `@register`'d classes:

```python
_ROBOT_SPECS_DIR = Path(os.environ.get("ROBOT_SPECS_DIR", str(_REPO_ROOT / "robot_specs")))
load_specs_from_dir(_ROBOT_SPECS_DIR)   # mlcore — registers RobotSpec instances
register_from_yaml(_ROBOT_SPECS_DIR)    # this repo — registers EnvAdapter factories
```

`ROBOT_SPECS_DIR` is read directly via `os.environ.get` (not Hydra) because `registrations.py` runs at import time, before Hydra composes its config. This is the documented exception in the workspace `CLAUDE.md` ("Hydra ou env-var checks").

`cube_reach_v1` is migrated to `robot_specs/cube_reach_v1.yaml`. The legacy `mlcore.robots.specs.cube_reach_v1` Python module and `configs/{env,dataset}/cube_reach_v1.yaml` are **kept** for now (coexistence, reversible) — YAML registration runs after the legacy `@register("cube_reach_v1") class CubeReachV1Adapter`, so it wins, logging an `EnvAdapterRegistry: overwriting existing adapter 'cube_reach_v1'` warning. A `cube_reach_v2` stub (`parent_id: cube_reach_v1`, `adapter: null`) is added as a lineage-tree fixture for the orchestrator's P1 work.

## Alternatives considered

1. **Orchestrator generates the same 6 files via subprocess** (extend `add_robot.py` as a CLI the orchestrator shells out to). Rejected: the orchestrator's `api` service has no writable mount into this repo's `src/` or `configs/`; only a new, narrowly-scoped data directory can be made writable without weakening the read-only sibling-repo guarantee.
2. **One YAML per robot, but keep `registrations.py` as the single source of adapter classes** (YAML only feeds `RobotSpec`, not `EnvAdapterRegistry`). Rejected: still requires a hand-written adapter class per robot, defeating the goal of "declare a robot from the frontend with no code changes."
3. **Support arbitrary `adapter.type` values generically** (plugin-style adapter resolution from YAML). Rejected as premature: only `mujoco_playground` adapters are constructible from `(env_name, task_description)` alone; hardware adapters need driver-specific wiring that cannot be expressed safely as data. Revisit if/when a second data-constructible adapter type appears.

## Consequences

**Positive**:
- New MuJoCo Playground robots (and lineage branches of existing ones) can be registered by adding a YAML file to `robot_specs/`, with zero Python source changes — the precondition for the orchestrator's `/api/v1/robots` create/branch endpoints (P1).
- `task` fields ("objectifs": `task_description`, `fps`, `episode_length`, `seed`, `success_threshold`, `max_episode_steps`) live in the same file as the spec and adapter — one artifact per robot version.
- `id` (technical key) is unchanged for the collect/train/eval pipeline — `cube_reach_v1` resolves identically whether the YAML or the legacy Python registration is the active one.

**Negative**:
- Two registration paths coexist for `cube_reach_v1` until P2 cleanup removes the legacy files (`mlcore/src/mlcore/robots/specs/cube_reach_v1.py`, `configs/env/cube_reach_v1.yaml`, `configs/dataset/cube_reach_v1.yaml`, the `@register("cube_reach_v1") class CubeReachV1Adapter` block). The overwrite warning makes the precedence visible but it's still two sources of truth in the interim.
- `tests/test_registrations.py`'s nomenclature guardian test (`test_class_name_matches_key`) now skips data-driven registry entries (`functools.partial`, not a `type`) — the snake_case ↔ PascalCase convention only applies to hand-written `@register`'d classes.
- `add_robot.py`'s `_build_spec` required a fix: it previously hardcoded `target_pos_key="cube_pos"` regardless of `--obs-keys`, which violates the new `target_pos_key in obs_keys` invariant (ml-core ADR-001) when `--obs-keys` is given without `cube_pos`. Now falls back to the last `--obs-keys` entry when `cube_pos` isn't present and `--target-pos-key` isn't given.

**Verification**: `pytest tests/test_yaml_registrations.py` (register_from_yaml: mujoco_playground factory, no-adapter skip, unsupported-type warning) + full `pytest -m unit` (79 passed, 1 skipped). Smoke: `python collect.py episodes=2 dataset.root=/tmp/smoke env=cube_reach_v1` resolves the YAML-registered factory and reaches `validate_spec_against_env` identically to the legacy class-based path (verified by comparing against `ROBOT_SPECS_DIR=/nonexistent` — both fail at the same point, a pre-existing CUDA/JAX backend limitation on this dev machine, unrelated to this change).
