# Architecture — lerobot-playground-portfolio

End-to-end imitation learning portfolio on MuJoCo Playground. This repo is the consumer side
of the platform: it depends on `robotics-platform-template` (HAL + env adapter contracts) and
`ml-core` (policy/trainer/evaluator implementations), and adds task-specific code (custom env,
demo collector, Hydra configs, train/eval entrypoints, visualisation).

## Pipeline overview

```
env (CubeReachV1)
  └─> scripted policy + DemoCollector
        └─> LeRobotDataset v3.0  (Parquet + MP4)
              └─> Trainer (Hydra + MLflow + HF Hub push)
                    └─> Policy checkpoint (ACT or Diffusion)
                          └─> Evaluator (N=50 × 3 seeds, bootstrap CI 95%)
```

Every step is Hydra-configurable and reproducible from `configs/`.

## Layout

```
lerobot-playground-portfolio/
├── src/playground/
│   ├── envs/         cube_reach_v1.py — Franka Panda dense reach reward
│   ├── data/         pipeline.py — ScriptedPolicy, DemoCollector, Episode
│   ├── policies/     act_wrapper.py, diffusion_wrapper.py (re-export from ml-core)
│   ├── training/     trainer.py — re-export from ml-core (MLflow + checkpoints + HF push)
│   ├── eval/         evaluator.py — re-export from ml-core + EvaluatorWithViz
│   ├── utils/        visualizer.py — render_episode, plot_rewards
│   └── scripts/      add_robot.py
├── train.py          Hydra entrypoint — dataset + policy + Trainer wiring
├── eval.py           Hydra entrypoint — checkpoint load + Evaluator
└── configs/
    ├── env/          cube_reach_v1.yaml
    ├── policy/       act.yaml, diffusion.yaml
    ├── training/     default.yaml, kaggle.yaml
    └── eval/         default.yaml
```

## Modules — what each provides

### `playground.envs`

| File | Provides |
|---|---|
| `cube_reach_v1.py` | `CubeReachV1` — subclasses `PandaPickCube`, dense reach reward, 200-step episodes, success at `dist < 0.05 m` |
| `cube_reach_v2.py` | v2 variant (forward-looking) |

Registered with `mujoco_playground` via `register_environment`.

### `playground.data`

| File | Provides |
|---|---|
| `pipeline.py` | `ScriptedPolicy`, `DemoCollector`, `Episode`, 16-dim state builder (ee 3 + cube 3 + joints 7 + delta 3) |

Output target: `LeRobotDataset` v3.0 (multi-episode Parquet + MP4) — repo_id `mefiezvous/cube-reach-v1-dataset`.

### `playground.policies`

Thin wrappers around `ml-core` policies — keep the public surface stable across this repo while
algorithms evolve upstream.

| Wrapper | Re-exports from | Action chunk |
|---|---|---|
| `ACTWrapper` | `mlcore.policies.act_wrapper.ACTWrapper` | `n_action_steps=100`, `n_obs_steps=1` |
| `DiffusionWrapper` | `mlcore.policies.diffusion_wrapper.DiffusionWrapper` | `n_action_steps=8`, `n_obs_steps=2`, `num_inference_steps=10` |

Both expose `state` 16-dim → `action` 8-dim (7 joint targets + 1 gripper).

### `playground.training`

| File | Provides |
|---|---|
| `trainer.py` | Re-export `Trainer` from `mlcore.training`. MLflow logging, periodic checkpointing, HF Hub push (`hf_repo_id` config), local checkpoint retention (`keep_last_n`) |

### `playground.eval`

| File | Provides |
|---|---|
| `evaluator.py` | Re-export `Evaluator` + `EvalResult` from `mlcore.eval`. Adds `EvaluatorWithViz` (saves MP4 + reward plot). Default N=50 episodes, bootstrap CI 95% on success rate |

### `playground.utils`

| File | Provides |
|---|---|
| `visualizer.py` | `EpisodeFrames` dataclass, `render_episode()` → MP4, `plot_rewards()` → PNG |

## Entrypoints

```bash
# Training — composes env + policy + training + logging configs
python train.py --config-name training/kaggle policy=act logging.run_name=...

# Evaluation — loads checkpoint, writes eval_report.json
python eval.py +eval.checkpoint_path=... +eval.n_episodes=50 logging.run_name=...
```

`train.py` builds the policy from `cfg.policy.name` (`act` / `diffusion`), instantiates
`LeRobotDataset` with `delta_timestamps` derived from `n_obs_steps` and `chunk_size`, then runs
`Trainer.train()`.

## Upstream dependencies

| Layer | Repo | Imports used |
|---|---|---|
| HAL | `robotics-platform-template` | `robotics_platform.hal.*`, `robotics_platform.envs.*` |
| Algorithms | `ml-core` | `mlcore.policies.*`, `mlcore.training.Trainer`, `mlcore.eval.Evaluator` |

Both are editable path dependencies declared in `pyproject.toml` under `[tool.uv.sources]`.

## Reproducibility

| Lever | Where |
|---|---|
| Fixed seeds | `cfg.training.seed`, `cfg.env.seed` |
| Versioned configs | `configs/` — committed Hydra YAML |
| Dataset hash | Logged to MLflow on every run |
| Full Hydra config snapshot | Logged to MLflow as artefact + printed at startup |
| HF Hub artefacts | Datasets and models published under `mefiezvous/cube-reach-v1-*` |

## Hydra config composition

```yaml
# configs/training/default.yaml
defaults:
  - env: cube_reach_v1
  - policy: act
  - eval: default
  - _self_
```

Override any field on the CLI: `policy=diffusion`, `training.batch_size=128`,
`training.device=cuda`, `logging.run_name=...`.
