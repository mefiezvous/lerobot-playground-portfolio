<!--
SPDX-FileCopyrightText: 2026 Arthur Mouraud
SPDX-License-Identifier: Apache-2.0
-->
# Security Policy

## Supported Versions
The `main` branch is the only supported version.

## Reporting a Vulnerability
Private reporting via GitHub Security Advisories: https://github.com/mefiezvous/lerobot-playground-portfolio/security/advisories/new

Expected first response: 7 days. Coordinated disclosure window: 90 days.

## Threat Model
- Public Apache-2.0 portfolio. Outbound only — no inbound listeners.
- Egress allowlist: huggingface.co, kaggle.com, wandb.ai.
- IP boundary: `_private/my-robot-stack` and proprietary stacks MUST NOT appear here. Anti-leak hook + CI job enforce this.

## Recommended
Enable GitHub push protection (Settings → Code security → Secret scanning push protection).
