# Status — lerobot-playground-portfolio
_Dernière mise à jour : 2026-05-20 · Session 2_

## Branche active
`feat/cube-reach-env` → PR #1 ouverte

## Ce qui est fait

### Infrastructure (Session 1)
- [x] Repo git + Apache-2.0 + SPDX headers
- [x] Hydra configs : `env/cube_reach_v1`, `policy/act`, `policy/diffusion`, `training/default`, `training/kaggle`
- [x] Pre-commit hooks : ruff, mypy, anti-leak
- [x] CI GitHub Actions : lint + typecheck + tests + anti-leak
- [x] CLAUDE.md, ARCHITECTURE.md, AGENTS.md, IP_STRATEGY.md
- [x] Skills Claude Code : `lerobot-patterns`, `mujoco-playground`, `kaggle-deployment`
- [x] Commands Claude Code : `/new-policy`, `/train-on-kaggle`, `/ip-check`
- [x] Repo GitHub public créé + remote configuré
- [x] Makefile

### Foundation ML (Session 2)
- [x] `src/playground/envs/cube_reach_v1.py` — CubeReachV1 (subclasse PandaPickCube, dense reach reward)
- [x] `src/playground/data/pipeline.py` — ScriptedPolicy (P-controller) + DemoCollector (→ LeRobotDataset v3.0)
- [x] `tests/conftest.py` — mock mujoco_playground pour tests CPU
- [x] `tests/test_env.py` — 7 tests unit + 4 integration
- [x] `tests/test_pipeline.py` — 10 tests unit
- [x] `pyproject.toml` — mypy overrides pour libs non-typées
- [x] `.github/workflows/ci.yaml` — `MUJOCO_PLAYGROUND_BACKEND=numpy`

## Ce qui reste

### Session 3 — Policy wrappers + train.py (priorité haute)
- [ ] `src/playground/policies/act_wrapper.py` — thin wrapper autour de `ACTPolicy` LeRobot
- [ ] `src/playground/policies/diffusion_wrapper.py` — thin wrapper autour de `DiffusionPolicy` LeRobot
- [ ] `src/playground/training/trainer.py` — boucle training Hydra-configurable
- [ ] `train.py` (racine) — entrypoint CLI : charge env + dataset + policy, MLflow, checkpoint resume
- [ ] Tests correspondants

### Session 4 — Eval + Notebooks
- [ ] `src/playground/eval/evaluator.py` — N=50 rollouts × 3 seeds, CI-95% bootstrap
- [ ] `eval.py` (racine) — entrypoint CLI : output `eval_report.json` + `eval_report.html`
- [ ] `notebooks/01_quickstart.ipynb` — rollouts aléatoires + eval pretrained (Colab)
- [ ] `notebooks/02_train_kaggle.ipynb` — training 2×T4 + resume + push Hub (Kaggle)
- [ ] `notebooks/03_evaluate.ipynb` — benchmark complet + CI plot
- [ ] `notebooks/04_publish_to_hub.ipynb` — dataset + model cards, Hub public uniquement

### HuggingFace Hub (après training)
- [ ] Dataset : `mefiezvous/cube-reach-v1-dataset`
- [ ] Modèle ACT : `mefiezvous/cube-reach-v1-act`
- [ ] Modèle Diffusion : `mefiezvous/cube-reach-v1-diffusion`

## Stack vérifiée
```
lerobot==0.5.1 · mujoco==3.5.0 · mujoco-mjx==3.8.0
mujoco_playground==0.2.0 · stable-baselines3==2.8.0
hydra-core==1.3.2 · mlflow==3.12.0 · peft==0.19.1 · loguru>=0.7
```
