<!--
SPDX-FileCopyrightText: 2026 Arthur Mouraud
SPDX-License-Identifier: Apache-2.0
-->

# Architecture Decision Records (ADR) — lerobot-playground-portfolio

Cross-module architectural decisions that affect this repo. Each ADR is a short, immutable note: once accepted, it is amended only via a successor ADR.

## When to write an ADR
- A decision changes the pipeline contract (CLI shape, env registration, policy interface).
- A decision removes or replaces an abstraction.
- A trade-off has been made between two viable alternatives.

If the decision is fully local and reversible, a code comment is enough.

## Format

`ADR-NNN-short-kebab-title.md`:

```markdown
# ADR-NNN — Title

- **Status**: Proposed | Accepted YYYY-MM-DD | Superseded by ADR-MMM | Implemented YYYY-MM-DD
- **Deciders**: Arthur Mouraud
- **Scope**: <which repos / modules>

## Context
## Decision
## Alternatives considered
## Consequences
```

## Related ADRs in sibling repos

- [robotics-platform-template/docs/adr/ADR-001](../../../robotics-platform-template/docs/adr/ADR-001-unify-on-envadapter.md) — Unify on EnvAdapter. The pipeline already follows this — no migration required in this repo. The nomenclature guardian test in [tests/test_registrations.py](../../tests/test_registrations.py) enforces the snake_case ↔ PascalCase convention from the template's GLOSSARY.

## Index

_(No local ADRs yet.)_
