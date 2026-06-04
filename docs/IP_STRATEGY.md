# IP Strategy — lerobot-playground-portfolio

This repository is **public**, Apache-2.0. It sits at the consumer end of a three-layer
robotics platform. This document spells out what can live here, what cannot, and how the
boundary is enforced.

## Three-layer architecture

| Layer | Visibility | License | Role |
|---|---|---|---|
| TEMPLATE (`robotics-platform-template`) | PUBLIC | Apache-2.0 | Generic HAL + env adapter Protocols |
| ML-CORE (`ml-core`) | shared lib | Apache-2.0 | Reusable policy / trainer / evaluator implementations |
| PUBLIC (this repo) | PUBLIC | Apache-2.0 | Task-specific env, configs, train/eval entrypoints, notebooks |
| Private layer | local only | Proprietary | Hardware adapters, customer-specific configs, never pushed |

The private layer exists in the local workspace only and is **never** referenced from any
public repo. Its existence is acknowledged abstractly here; nothing more.

## Why this repo is PUBLIC Apache-2.0

- Portfolio-facing — open science, open code, reproducible benchmarks.
- All assets (env, policies, datasets, models) are generic and shareable.
- HuggingFace Hub artefacts (`mefiezvous/cube-reach-v1-*`) are public datasets / model weights.
- Apache-2.0 maximises adoption while preserving attribution.

## What MUST NOT appear here

- References to local-only directories or proprietary stacks.
- License markers `LicenseRef-Proprietary` or `All Rights Reserved`.
- Hardware-specific code (specific robot brands, model numbers, calibration data).
- Customer or engagement names.
- Fine-tuned weights trained on proprietary data.
- Production safety configs tied to a specific deployment.

Generic robot models (Franka Panda) and standard simulators (MuJoCo Playground) are fine —
they are themselves open and well-known.

## Anti-leak enforcement

A pre-commit hook (also re-run in CI) greps every staged change for the forbidden patterns
listed above and verifies the SPDX header is present on every `.py`. The hook is **active**
and must not be bypassed with `--no-verify`. If it fires, fix the content — do not silence it.

> **Important — best-effort only (LRB-005):** The hook uses case-insensitive fixed-string
> matching over a finite pattern list. It will not catch URL-encoded paths, base64-encoded
> blobs, Unicode homoglyphs, or patterns split across diff context lines. It is a helpful
> early warning, not a complete security control. The primary control is developer discipline
> and code review. If in doubt, run `make audit` and inspect the diff manually before pushing.

| Check | Action on FAIL |
|---|---|
| Forbidden string match (proprietary markers, private-layer paths) | Commit rejected |
| Missing SPDX header on a new `.py` | Commit rejected |
| `make audit` | Returns non-zero — blocks release |

Run `make audit` (or `/ip-check`) before any HuggingFace Hub push or release tag.

## Reporting

Any leak — even retrospective — should be treated as a security incident: remove the file
from history (`git filter-repo`), force-push if necessary, and rotate any exposed credentials.
The repo owner (Arthur Mouraud) is the point of contact.
