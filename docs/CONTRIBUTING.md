# Contributing — lerobot-playground-portfolio

This repo is a public Apache-2.0 portfolio. Contributions must preserve its open-source status,
its IP boundaries, and its reproducibility discipline.

## Strict rules

1. **No private-layer content.** No references to local-only stacks, proprietary projects, or
   any directory outside the public workspace tree.
2. **No proprietary license markers.** No `LicenseRef-Proprietary`, no `All Rights Reserved`.
   This repo is Apache-2.0 only.
3. **No hardware-specific or production safety code.** This is a generic playground portfolio —
   anything tied to a specific robot deployment belongs elsewhere.
4. **SPDX header** on every `.py`:
   ```
   # SPDX-FileCopyrightText: 2026 Arthur Mouraud
   # SPDX-License-Identifier: Apache-2.0
   ```
5. **Anti-leak pre-commit hook** is active. It blocks the patterns above. Never bypass with
   `--no-verify` — fix the issue instead.

## Workflow

- Conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`
- Feature branches only — no direct commits to `main`.
- PR required (even solo). CI must be green before merge.
- TDD: write the test first, then the implementation.
- Plan mode (Claude Code) for any change touching 3+ files.

## Local checks

```bash
make install     # uv sync + pre-commit install
make lint        # ruff check + format
make typecheck   # mypy strict
make test        # pytest -m "not gpu" --cov-fail-under=70
make test-all    # pytest (GPU required)
make audit       # anti-leak scan
```

Pre-commit hooks (`ruff`, `mypy`, `anti-leak`, SPDX) run on every commit.

## Code standards

- Type hints everywhere; `mypy --strict` clean.
- `from loguru import logger` — never `print()`.
- No direct `os.environ` access — Hydra config or explicit env-var helper.
- All paths via `pathlib.Path`, never raw strings.
- Google-style docstrings on every public API.
- Pytest markers: `@pytest.mark.unit` (default), `@pytest.mark.integration`, `@pytest.mark.gpu`.
- Coverage gate: 70% global, 100% on `robotics_platform.hal`.

## IP audit before any HF Hub push or release

Run `make audit` (or the `/ip-check` Claude Code command). It greps for forbidden patterns and
verifies SPDX headers. Block release on FAIL — no exceptions.

## What does NOT belong here

Anything tied to a specific robot deployment, customer engagement, or proprietary dataset.
Generic HAL contracts belong in `robotics-platform-template`. Reusable algorithm code belongs
in `ml-core`. This repo is for the portfolio-facing task/env/notebook layer only.
