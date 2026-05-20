# lerobot-playground-portfolio

[![CI](https://github.com/mefiezvous/lerobot-playground-portfolio/actions/workflows/ci.yaml/badge.svg)](https://github.com/mefiezvous/lerobot-playground-portfolio/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![HF Dataset](https://img.shields.io/badge/HF-dataset-orange.svg)](https://huggingface.co/datasets/mefiezvous/cube-reach-v1-dataset)

> End-to-end robotics portfolio: custom MuJoCo Playground environments, ACT and Diffusion Policy
> training, and benchmark evaluation — all running on free-tier Kaggle GPUs (2×T4).

## Results

| Policy | Environment | Success Rate | CI 95% |
|--------|-------------|-------------|--------|
| ACT | CubeReachV1 | TBD | TBD |
| Diffusion Policy | CubeReachV1 | TBD | TBD |

## Quickstart

```bash
git clone https://github.com/mefiezvous/lerobot-playground-portfolio
cd lerobot-playground-portfolio
uv sync
uv run python -c "from playground.envs import CubeReachV1; print('OK')"
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/01_quickstart.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](notebooks/02_train_kaggle.ipynb)

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md). HAL layer from [robotics-platform-template](https://github.com/mefiezvous/robotics-platform-template).

## Reproducibility

All experiments use fixed seeds. Dataset hash and config are logged with every run via MLflow.
Configs are versioned in `configs/`. Notebooks are self-contained with install cells.

## Roadmap

- [ ] SmolVLA fine-tuning with PEFT LoRA
- [ ] Sim-to-real transfer evaluation
- [ ] Additional MuJoCo Playground environments
- [ ] Multi-task training
