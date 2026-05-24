# Roadmap — lerobot-playground-portfolio

Forward-looking only. For volatile state (current branch, in-progress fixes, P0 list), see workspace memory.

## Current release: v0.1.0 (in preparation)

End-to-end portfolio with custom MuJoCo Playground env (CubeReachV1), ACT and Diffusion Policy wrappers, Hydra-configured training/eval, Kaggle 2×T4 notebook pipeline. Code complete, awaiting first real training run on Kaggle.

## Stable surface (v0.1)

- `playground.envs.cube_reach_v1.CubeReachV1` — custom MuJoCo Playground env (dense reach reward, 200-step episodes)
- `playground.data.pipeline` — `ScriptedPolicy`, `DemoCollector`, `Episode` (LeRobotDataset v3.0 export)
- `playground.policies.act_wrapper.ACTWrapper` (re-export from `mlcore`)
- `playground.policies.diffusion_wrapper.DiffusionWrapper` (re-export from `mlcore`)
- `playground.training.trainer.Trainer` (re-export from `mlcore`, with MLflow, checkpoint resume, optional HF Hub push)
- `playground.eval.evaluator.Evaluator` + `EvalResult` (re-export from `mlcore`)
- `playground.utils.visualizer` — `render_episode()`, `plot_rewards()`, `EvaluatorWithViz`
- CLI: `train.py`, `eval.py`, `scripts/add_robot.py`
- Notebooks: `01_quickstart`, `02_train_kaggle`, `03_evaluate`, `04_publish_hf`

## Pending for v0.1.0 release

| Item | Where |
|---|---|
| First successful Kaggle training run (ACT) | `notebooks/02_train_kaggle.ipynb` |
| Publish dataset `mefiezvous/cube-reach-v1-dataset` | `notebooks/04_publish_hf.ipynb` |
| Publish models `mefiezvous/cube-reach-v1-act`, `mefiezvous/cube-reach-v1-diffusion` | `notebooks/04_publish_hf.ipynb` |
| Fill in README.md `Results` table (success rate + CI 95%) | `README.md` |
| Demo GIF in README | `README.md` |
| Release tag v0.1.0 | git |

## v0.2 candidates (not committed)

| Item | Rationale |
|---|---|
| SmolVLA fine-tuning with PEFT LoRA | Showcase parameter-efficient adaptation |
| Sim-to-real transfer evaluation | Bridge to private hardware layer |
| Additional MuJoCo Playground environments | Multi-task showcase |
| Multi-task training | Single policy across envs |
| Vectorized rollouts in `Evaluator` | Speed up N=50×3-seed eval |

## Maintenance discipline

- IP boundaries: never reference `_private/`, `my-robot-stack`, or proprietary content.
- SPDX header on every `.py`. anti-leak pre-commit + CI enforce this.
- Coverage gate: 70% global, 100% on platform.hal (from template).
- All sessions logged in workspace memory (`project_state.md`), not duplicated here.

## Dependencies

| Layer | Repo | Import path |
|---|---|---|
| HAL | `robotics-platform-template` | `robotics_platform.hal.*`, `robotics_platform.envs.*` |
| Algorithms | `ml-core` | `mlcore.policies.*`, `mlcore.training.*`, `mlcore.eval.*`, `mlcore.robots.*` |

Both are editable path dependencies in `pyproject.toml`.
