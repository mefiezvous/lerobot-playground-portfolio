# lerobot-playground-portfolio

[![CI](https://github.com/mefiezvous/lerobot-playground-portfolio/actions/workflows/ci.yaml/badge.svg)](https://github.com/mefiezvous/lerobot-playground-portfolio/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![HF Dataset](https://img.shields.io/badge/HF-dataset-orange.svg)](https://huggingface.co/datasets/mefiezvous/cube-reach-v1-dataset)

> End-to-end robotics portfolio: custom MuJoCo Playground environments, ACT and Diffusion Policy
> training, and benchmark evaluation — all running on free-tier Kaggle GPUs (2×T4).

## Quickstart

```bash
git clone https://github.com/mefiezvous/lerobot-playground-portfolio
cd lerobot-playground-portfolio
uv sync
uv run python -c "from playground.envs.cube_reach_v1 import CubeReachV1; print('OK')"
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/01_quickstart.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](notebooks/02_train_kaggle.ipynb)

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/01_quickstart.ipynb` | Local collect-demo smoke test (CPU, ~30 s) |
| `notebooks/02_train_kaggle.ipynb` | Train ACT / Diffusion on Kaggle 2×T4 (~4 h) |
| `notebooks/03_evaluate.ipynb` | Load checkpoint, run `Evaluator`, plot rewards |
| `notebooks/04_publish_hf.ipynb` | Push dataset + models to HuggingFace Hub |

## Train / Eval CLI

```bash
# Train ACT on Kaggle GPU config
python train.py --config-name training/kaggle policy=act logging.run_name=kaggle_act_001

# Evaluate a checkpoint
python eval.py +eval.checkpoint_path=checkpoints/cube_reach_v1/act/checkpoint_00010000.ckpt
```

## Results

| Policy | Environment | Success Rate | CI 95% |
|---|---|---|---|
| ACT | CubeReachV1 | TBD | TBD |
| Diffusion Policy | CubeReachV1 | TBD | TBD |

> Awaiting first Kaggle training run. See [docs/ROADMAP.md](docs/ROADMAP.md).

## Reproducibility

Fixed seeds. Dataset hash + Hydra config logged via MLflow on every run. Configs versioned in `configs/`. Notebooks are self-contained with install cells.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 3-layer flow, modules, ADRs
- [docs/ROADMAP.md](docs/ROADMAP.md) — current release status + v0.2 candidates
- [docs/AGENTS.md](docs/AGENTS.md) — sub-agents
- [docs/IP_STRATEGY.md](docs/IP_STRATEGY.md) — layer separation
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — workflow, code standards

## License

Apache-2.0. Copyright 2026 Arthur Mouraud. HAL from [robotics-platform-template](https://github.com/mefiezvous/robotics-platform-template).
