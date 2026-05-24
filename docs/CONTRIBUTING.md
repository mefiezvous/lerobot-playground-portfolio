# Contributing — lerobot-playground-portfolio

This is a public Apache-2.0 portfolio project. Contributions must respect IP boundaries (no proprietary content) and the established workflow.

## IP rules (non-negotiable)

1. NEVER reference `_private/`, `my-robot-stack/`, or any proprietary content.
2. NEVER use `LicenseRef-Proprietary` or `All Rights Reserved`.
3. NEVER add hardware-specific code or production safety configs — those live in the private layer.
4. SPDX header at top of every `.py`:
   ```
   # SPDX-FileCopyrightText: 2026 Arthur Mouraud
   # SPDX-License-Identifier: Apache-2.0
   ```

The `anti-leak` pre-commit hook and CI both block patterns above.

## Code standards

- Type hints everywhere; `mypy --strict` clean.
- `from loguru import logger` — never `print()`.
- No direct `os.environ` access — Hydra cfg or explicit env-var checks.
- Imports order: stdlib → third-party → local (blank line between groups).
- Google-style docstrings on public API.
- Pytest markers: `@pytest.mark.unit` (default), `@pytest.mark.integration`, `@pytest.mark.gpu`.
- Coverage gate: 70% global, 100% on `robotics_platform.hal` (inherited from template).

## Workflow

- **Plan mode** for any change touching 3+ files.
- **TDD**: write test first, then implementation.
- Conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`.
- Feature branches only — no direct commits to `main`.
- PR required (even solo) — portfolio signal.

## Local checks

```bash
make install     # uv sync + pre-commit install
make lint        # ruff check + format
make typecheck   # mypy strict
make test        # pytest -m "not gpu" --cov-fail-under=70
make test-all    # pytest (GPU required)
make audit       # anti-leak scan
```

## Available skills & sub-agents

- Skills (invoke by name): `lerobot-patterns`, `mujoco-playground`, `kaggle-deployment`
- Sub-agents: see [AGENTS.md](AGENTS.md) — `lerobot-expert`, `eval-runner`, `ip-guardian`
- Commands: `/new-policy`, `/train-on-kaggle`, `/ip-check`

## IP audit before any HF Hub push or release

Run `make audit` (or `/ip-check`). It greps for the forbidden patterns above and verifies SPDX headers. Block release on FAIL — no exceptions.
