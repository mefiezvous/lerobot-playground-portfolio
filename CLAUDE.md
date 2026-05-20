# Claude Code — lerobot-playground-portfolio

## Project Identity
Public portfolio: LeRobot + MuJoCo Playground on free-tier Kaggle GPUs (2×T4).
License: Apache-2.0. Python 3.12+. Author: Arthur Mouraud (mefiezvous).

## Stack
- lerobot==0.5.1 (LeRobotDataset v3.0, ACT, DiffusionPolicy)
- mujoco==3.5.0 + mujoco-mjx==3.8.0 + mujoco_playground==0.2.0 (Warp backend)
- stable-baselines3==2.8.0, hydra-core==1.3.2, mlflow==3.12.0, peft==0.19.1, loguru>=0.7
- HAL layer: robotics-platform-template (local path dep at ../../robotics-platform-template)

## Absolute Rules (never violate)
1. NEVER reference `_private/`, `my-robot-stack/`, or any proprietary content
2. NEVER use `LicenseRef-Proprietary` or `All Rights Reserved` in any file
3. SPDX header required at top of every .py file:
   `# SPDX-FileCopyrightText: 2026 Arthur Mouraud`
   `# SPDX-License-Identifier: Apache-2.0`
4. No hardcoded values — use Hydra configs or pathlib.Path
5. No print() — use loguru: `from loguru import logger`
6. No os.environ direct access — gate secrets via Hydra or env-var checks

## Workflow
- Plan mode for any change touching 3+ files
- TDD: write test first, then implementation
- Conventional commits: feat:, fix:, docs:, chore:, refactor:, test:
- Feature branches only — no direct commits to main
- PRs required even when working solo (portfolio signal)

## Code Standards
- Type hints everywhere, mypy strict
- Google-style docstrings for public API
- Imports: stdlib → third-party → local (blank line between groups)
- Coverage: 70% global minimum, 100% on platform.hal (from template)
- Pytest markers: @pytest.mark.unit (default), @pytest.mark.integration, @pytest.mark.gpu

## Available Skills (invoke by name)
- `lerobot-patterns` — LeRobotDataset v3.0, policy loading, Hub patterns
- `mujoco-playground` — custom env registration, Warp backend, vectorized rollouts
- `kaggle-deployment` — 2×T4 setup, 9h resume, Kaggle secrets, notebook patterns

## Available Commands
- `/new-policy` — scaffold a new policy wrapper
- `/train-on-kaggle` — generate Kaggle notebook for a training run
- `/ip-check` — run ip-guardian audit on staged files

## Architecture Pointer
See ARCHITECTURE.md. HAL interfaces live in robotics-platform-template.
Never add hardware-specific code here — it goes in the private layer.
