# Architecture

## Three-Layer Platform

```
robotics-workspace/                  # local workspace (not a git repo)
├── lerobot-playground-portfolio/    # this repo (PUBLIC, Apache-2.0)
├── robotics-platform-template/      # HAL layer (TEMPLATE, Apache-2.0)
└── _private/my-robot-stack/         # proprietary stubs (PRIVATE, local only)
```

## Dependency Flow

```
PRIVATE ──imports──> TEMPLATE (HAL)
PUBLIC  ──imports──> TEMPLATE (HAL)
PUBLIC  never imports PRIVATE
```

## Key Components

| Module | Purpose |
|--------|---------|
| `src/playground/envs/` | Custom MuJoCo Playground environments |
| `src/playground/policies/` | Thin wrappers around LeRobot policies |
| `src/playground/data/` | Trajectory collection → LeRobotDataset v3.0 |
| `train.py` | Hydra-configured training entry point |
| `eval.py` | Benchmark suite (N=50 rollouts, 3 seeds, CI-95) |

## ADRs

### ADR-001: LeRobotDataset v3.0
Using v3.0 format (multi-episode Parquet + MP4). Reason: more efficient than v2, supported by lerobot 0.5.1.

### ADR-002: MuJoCo Warp Backend
Using `mujoco_playground` with Warp (JAX+GPU) backend. Reason: fastest on T4, native to mujoco_playground 0.2.0.

### ADR-003: Hydra Config
Using Hydra 1.3 for config composition. Reason: CLI overrides for Kaggle param sweeps, composable env/policy/training configs.

### ADR-004: CubeReachV1 Environment
Subclasses `PandaPickCube` with dense reach-only reward. Reason: simple enough to train on free-tier GPU in one session.
