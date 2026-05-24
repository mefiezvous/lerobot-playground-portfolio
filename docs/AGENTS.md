# Agents — lerobot-playground-portfolio

Claude Code skills, slash commands, and sub-agent conventions used in this repo. All assets
live under `.claude/` and are checked into the repo (no machine-local state).

## Skills

Reusable domain knowledge bundles in `.claude/skills/<name>/SKILL.md`. Invoked by Claude Code
when a task matches the skill's trigger.

| Skill | Scope |
|---|---|
| `lerobot-patterns` | LeRobotDataset v3.0 format, policy loading, `push_to_hub`, `from_pretrained` |
| `mujoco-playground` | Custom env registration, Warp/JAX backend, vectorized rollouts |
| `kaggle-deployment` | 2×T4 setup, 9 h session resume, Kaggle secrets, notebook patterns |

Edit a skill by updating its `SKILL.md` — the trigger and behaviour sections drive Claude Code's
selection logic.

## Slash commands

Project-scoped prompts in `.claude/commands/<name>.md`. Invoked as `/<name>`.

| Command | Purpose |
|---|---|
| `/new-policy` | Scaffold a new policy wrapper (test-first, re-export from `ml-core`, Hydra config) |
| `/train-on-kaggle` | Walk through the Kaggle notebook pipeline (secrets, 2×T4, resume, HF push) |
| `/ip-check` | Run the anti-leak audit: forbidden patterns + SPDX header verification |

## Sub-agent conventions

This repo currently does not declare named sub-agents under `.claude/agents/`. When spawning
a sub-agent for a task, follow these conventions:

| Role | Trigger | Constraints |
|---|---|---|
| LeRobot expert | Tasks touching `LeRobotDataset`, ACT, Diffusion, SmolVLA, `push_to_hub`, `from_pretrained` | Use lerobot 0.5.1 API only; only push to public HF Hub repos under `mefiezvous/*` |
| Eval runner | Tasks touching `eval.py`, success-rate computation, benchmark reports | N≥50 rollouts × 3 seeds; bootstrap CI 95% (n_bootstrap=10000); fixed seeds logged to MLflow |
| IP guardian | Tasks touching licensing, layer boundaries, pre-commit hook, releases, Hub pushes | Run audit before release; block on FAIL — no exceptions |

These conventions are enforced through skills, commands, and the anti-leak pre-commit hook
rather than through explicit agent declarations.

## Configuration

`.claude/settings.json` declares the project-level Claude Code configuration (permissions,
allowed tools). Keep it minimal and reviewable — anything reflexive (auto-actions on every
turn) belongs in user-level settings, not the repo.
