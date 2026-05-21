# Status — lerobot-playground-portfolio
_Dernière mise à jour : 2026-05-21 · Session 5 (fixes architecturaux)_

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

### Policy wrappers + train.py (Session 3)
- [x] `src/playground/policies/act_wrapper.py` — thin wrapper autour de `ACTPolicy` LeRobot
- [x] `src/playground/policies/diffusion_wrapper.py` — thin wrapper autour de `DiffusionPolicy` LeRobot
- [x] `src/playground/training/trainer.py` — boucle training Hydra-configurable, MLflow, checkpoint resume
- [x] `train.py` (racine) — entrypoint CLI Hydra (ACT ou Diffusion, --config-name)
- [x] `tests/test_policies.py` — 4 smoke tests (re-exports → mlcore)
- [x] `tests/test_trainer.py` — 4 smoke tests (re-exports → mlcore)
- [x] `tests/test_train.py` — 3 smoke tests (_build_policy : act / diffusion / unknown)
- [x] Tests d'implémentation (ACT×8, Diffusion×5, Trainer×10) migrés dans ml-core
- [x] Fix `pyproject.toml` — chemin `../robotics-platform-template` (était `../../`) + hatch wheel config
- [x] Configs enrichies — `input_shapes`, `output_shapes`, `dataset.*`, `training.lr`

### Eval + Notebooks (Session 4) ✅
- [x] `src/playground/eval/evaluator.py` — `EvalResult` dataclass + `Evaluator` class (n_episodes, success_rate, mean_reward, mlflow logging)
- [x] `eval.py` (racine) — entrypoint CLI Hydra + `_run_eval()` (testable sans Hydra), écrit `eval_report.json`
- [x] `tests/test_eval.py` — 4 tests unit (runs_n_episodes, success_rate, mlflow logging, entrypoint smoke)
- [x] `notebooks/01_quickstart.ipynb` — env setup + ScriptedPolicy → 5 épisodes + DataFrame
- [x] `notebooks/02_train_kaggle.ipynb` — guide complet Kaggle 2×T4 : install, secrets, train, resume
- [x] `notebooks/03_evaluate.ipynb` — charge checkpoint → Evaluator → EvalResult + plot
- [x] `notebooks/04_publish_hf.ipynb` — HF login → push dataset + ACT + Diffusion

## Ce qui reste

### HuggingFace Hub (après training réel)
- [ ] Dataset : `mefiezvous/cube-reach-v1-dataset` — collecter avec DemoCollector + push
- [ ] Modèle ACT : `mefiezvous/cube-reach-v1-act` — entraîner sur Kaggle + push
- [ ] Modèle Diffusion : `mefiezvous/cube-reach-v1-diffusion` — entraîner sur Kaggle + push

### Post-v0.1.0 (optionnel)
- [ ] README.md — badges CI, demo GIF, lien Hub
- [ ] ARCHITECTURE.md — mise à jour avec eval layer
- [ ] Release tag v0.1.0

## Stack vérifiée
```
lerobot==0.5.1 · mujoco==3.5.0 · mujoco-mjx==3.8.0
mujoco_playground==0.2.0 · stable-baselines3==2.8.0
hydra-core==1.3.2 · mlflow==3.12.0 · peft==0.19.1 · loguru>=0.7
```
