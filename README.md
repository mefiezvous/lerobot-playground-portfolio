# lerobot-playground-portfolio

[![CI](https://github.com/mefiezvous/lerobot-playground-portfolio/actions/workflows/ci.yaml/badge.svg)](https://github.com/mefiezvous/lerobot-playground-portfolio/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

> LeRobot playground portfolio — imitation learning on MuJoCo Playground.
> ACT + Diffusion Policy on `CubeReachV1`, trained on free-tier Kaggle GPUs (2×T4).

## What this is

A small, end-to-end portfolio repo demonstrating:
- `playground.envs.cube_reach_v1.CubeReachV1` — custom MuJoCo Playground env (Franka Panda, dense reach reward)
- ACT + Diffusion Policy via thin wrappers around `ml-core` policy implementations
- Hydra-driven `train.py` / `eval.py` entrypoints with MLflow logging and checkpoint resume
- LeRobotDataset v3.0 demo pipeline (scripted policy → episodes → Parquet + MP4)
- HuggingFace Hub publishing for both datasets and trained models

Everything runs locally (CPU smoke test) or on a single Kaggle session (2×T4, ~4 h).

## Install

```bash
git clone https://github.com/mefiezvous/lerobot-playground-portfolio
cd lerobot-playground-portfolio
uv sync
```

Path dependencies on the HAL (`robotics-platform-template`) and algorithms (`ml-core`) layers
are resolved as editable installs via `[tool.uv.sources]`.

## Quickstart

```bash
# 1. Collect a few scripted demos (CPU, smoke test)
uv run python -c "from playground.envs.cube_reach_v1 import CubeReachV1; print('OK')"

# 2. Train ACT on the Kaggle profile (2×T4, 100 k steps)
uv run python train.py --config-name training/kaggle policy=act \
    logging.run_name=kaggle_act_001

# 3. Evaluate a checkpoint (N=50 rollouts)
uv run python eval.py +eval.checkpoint_path=checkpoints/checkpoint_00010000.ckpt \
    +eval.n_episodes=50 logging.run_name=eval_run_001
```

Reproducible by construction: fixed seeds, Hydra configs versioned in `configs/`,
dataset hash + full config logged to MLflow on every run.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — pipeline overview, module layout, dependencies
- [docs/ROADMAP.md](docs/ROADMAP.md) — forward-looking
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — workflow & strict rules
- [docs/AGENTS.md](docs/AGENTS.md) — Claude Code skills & commands for this repo
- [docs/IP_STRATEGY.md](docs/IP_STRATEGY.md) — layer separation, what does NOT belong here

## License

Apache-2.0. See [LICENSE](LICENSE). Copyright 2026 Arthur Mouraud.
HAL contracts from [robotics-platform-template](https://github.com/mefiezvous/robotics-platform-template).
