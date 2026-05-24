# CLAUDE.md — lerobot-playground-portfolio

## Identity
Public portfolio (Apache-2.0): imitation learning policies (ACT, Diffusion) on MuJoCo Playground,
trained on free-tier Kaggle GPUs. Python 3.12+. Author: Arthur Mouraud. Package name: `playground`.

## Critical Rules
1. NEVER reference private layers, proprietary stacks, or non-public projects.
2. NEVER use `LicenseRef-Proprietary` or `All Rights Reserved` — this repo is Apache-2.0 only.
3. SPDX header required at top of every `.py`:
   `# SPDX-FileCopyrightText: 2026 Arthur Mouraud`
   `# SPDX-License-Identifier: Apache-2.0`
4. Anti-leak pre-commit hook is active — never bypass with `--no-verify`.
5. All configuration goes through Hydra — no hardcoded paths or hyperparameters.
6. `from loguru import logger` — never `print()`. No direct `os.environ` access.

## Code Standards
- mypy strict, type hints everywhere.
- TDD, pytest markers `@pytest.mark.unit` / `integration` / `gpu`.
- Coverage gate: 70% global, 100% on `robotics_platform.hal` (inherited from template).
- Conventional commits, feature branches, PR required even solo.

## Documentation enfant
- [README.md](README.md) — purpose, install, quickstart
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — pipeline, modules, dependencies
- [docs/ROADMAP.md](docs/ROADMAP.md) — forward-looking
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — workflow & strict rules
- [docs/AGENTS.md](docs/AGENTS.md) — Claude Code skills & commands
- [docs/IP_STRATEGY.md](docs/IP_STRATEGY.md) — layer separation, scope

## Workspace context (non committé)
- Cross-repo rules & memory : `../CLAUDE.md` racine workspace
- État volatile (branche active, P0) : `memory/project_state.md`
