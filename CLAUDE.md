# CLAUDE.md — lerobot-playground-portfolio

## Identity
Public portfolio (Apache-2.0): LeRobot + MuJoCo Playground policies trained on free-tier Kaggle GPUs (2×T4). Python 3.12+. Author: Arthur Mouraud (mefiezvous).

## Stack
- `lerobot==0.5.1` · `mujoco==3.8.0` / `mujoco-mjx==3.8.0` · `mujoco_playground==0.2.0`
- `stable-baselines3==2.8.0` · `hydra-core==1.3.2` · `mlflow==3.12.0` · `peft==0.19.1`
- Path deps: `robotics-platform-template` (HAL), `ml-core` (algorithms)

## Critical Rules
1. NEVER reference `_private/`, `my-robot-stack/`, or proprietary content.
2. NEVER use `LicenseRef-Proprietary` or `All Rights Reserved`.
3. SPDX header on every `.py`:
   `# SPDX-FileCopyrightText: 2026 Arthur Mouraud` / `# SPDX-License-Identifier: Apache-2.0`
4. `from loguru import logger` — no `print()`.
5. No hardcoded values — Hydra configs or `pathlib.Path`.
6. No direct `os.environ` access.

## Code Standards
- mypy strict, type hints everywhere.
- TDD; pytest markers `@pytest.mark.unit` / `integration` / `gpu`.
- Coverage 70% global, 100% on `robotics_platform.hal` (template).
- Conventional commits, feature branches, PR even solo.

## Available Skills (invoke by name)
- `lerobot-patterns` — LeRobotDataset v3.0, policy loading, Hub patterns
- `mujoco-playground` — custom env registration, Warp backend, vectorized rollouts
- `kaggle-deployment` — 2×T4 setup, 9h resume, Kaggle secrets, notebook patterns

## Available Commands
- `/new-policy` · `/train-on-kaggle` · `/ip-check`

## Documentation enfant
- [README.md](README.md) — quickstart, results, badges
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 3-layer flow, modules, ADRs
- [docs/ROADMAP.md](docs/ROADMAP.md) — forward-looking
- [docs/AGENTS.md](docs/AGENTS.md) — sub-agents (lerobot-expert, eval-runner, ip-guardian)
- [docs/IP_STRATEGY.md](docs/IP_STRATEGY.md) — layer separation, scope
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — workflow, IP rules, code standards

## Workspace context (non committé)
- Cross-repo rules & memory : `../CLAUDE.md` racine workspace
- État volatile (branche active, P0, incohérences) : `memory/project_state.md`
