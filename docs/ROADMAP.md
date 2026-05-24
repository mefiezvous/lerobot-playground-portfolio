# Roadmap — lerobot-playground-portfolio

Forward-looking only. For volatile state (current branch, in-progress fixes), see workspace memory.

## Current state: v0.1.0 (in preparation)

End-to-end portfolio with custom MuJoCo Playground env (`CubeReachV1`), ACT and Diffusion Policy
wrappers, Hydra-configured training and evaluation, and a Kaggle 2×T4 notebook pipeline.
Code complete; awaiting first full Kaggle training run + HF Hub publish.

## Stable surface (v0.1)

- `playground.envs.cube_reach_v1.CubeReachV1` — custom env (dense reach reward, 200-step episodes)
- `playground.data.pipeline` — `ScriptedPolicy`, `DemoCollector`, `Episode` (LeRobotDataset v3.0)
- `playground.policies.act_wrapper.ACTWrapper` (re-export from `ml-core`)
- `playground.policies.diffusion_wrapper.DiffusionWrapper` (re-export from `ml-core`)
- `playground.training.trainer.Trainer` (re-export, MLflow + checkpoint resume + HF Hub push)
- `playground.eval.evaluator.Evaluator` + `EvalResult` (re-export from `ml-core`)
- `playground.utils.visualizer` — `render_episode()`, `plot_rewards()`, `EpisodeFrames`
- CLI: `train.py`, `eval.py`
- Configs: `configs/{env,policy,training,eval}/*.yaml`

## Pending for v0.1.0 release

| Item | Where |
|---|---|
| First successful Kaggle training run (ACT) | Kaggle 2×T4 |
| Publish dataset `mefiezvous/cube-reach-v1-dataset` | HF Hub |
| Publish models `mefiezvous/cube-reach-v1-act`, `mefiezvous/cube-reach-v1-diffusion` | HF Hub |
| Fill README results table (success rate + CI 95%) | `README.md` |
| Release tag `v0.1.0` | git |

## vNext candidates (not committed)

| Item | Rationale |
|---|---|
| LIBERO environment adapter | Standard benchmark suite, broaden showcase beyond CubeReach |
| SmolVLA fine-tuning with PEFT LoRA | Demonstrate parameter-efficient adaptation |
| Multi-task training (single policy, several envs) | Generalisation story |
| Vectorized rollouts in `Evaluator` | Speed up N=50 × 3-seed eval |
| Additional MuJoCo Playground envs (push, stack) | Multi-task showcase |

These are candidates — none planned until the v0.1 release is shipped and validated end-to-end.

## Maintenance discipline

- IP boundaries: no proprietary content, no private-layer references — enforced by anti-leak hook + CI.
- SPDX header on every `.py`. Apache-2.0 only — no exceptions.
- Coverage gate: 70% global, 100% on `robotics_platform.hal` (inherited from template).
- No breaking changes to the `playground.*` public surface within a minor version.
- All session state lives in workspace memory (`project_state.md`), not in this file.

## Upstream dependencies

| Layer | Repo | Import path |
|---|---|---|
| HAL | `robotics-platform-template` | `robotics_platform.hal.*`, `robotics_platform.envs.*` |
| Algorithms | `ml-core` | `mlcore.policies.*`, `mlcore.training.*`, `mlcore.eval.*` |

Changes upstream that affect these imports require a coordinated bump here.
