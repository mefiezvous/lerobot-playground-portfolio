# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
.PHONY: install lint typecheck test audit clean

install:
	uv sync --extra dev
	pre-commit install

lint:
	uv run ruff check src/ tests/ --fix
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/ -m "not gpu" --cov=src --cov-report=term-missing

test-all:
	uv run pytest tests/ --cov=src --cov-report=term-missing

audit:
	@bash .git-hooks/pre-commit-anti-leak.sh || true
	@for p in "_private/" "my-robot-stack/" "proprietary_" "LicenseRef-Proprietary" "All Rights Reserved"; do \
	  grep -r --include="*.py" --include="*.yaml" --include="*.md" "$$p" src/ configs/ && echo "LEAK: $$p" || true; \
	done
	@echo "Audit complete."

clean:
	rm -rf .venv dist *.egg-info .mypy_cache .ruff_cache .coverage htmlcov mlruns outputs multirun checkpoints
