# IP Strategy

This repository is PUBLIC under Apache-2.0 license.

## What This Repo Contains
- Generic MuJoCo Playground environments (no hardware-specific code)
- Thin wrappers around LeRobot standard policies (ACT, Diffusion Policy)
- Reproducible benchmark results on standard tasks
- Training/eval infrastructure (Hydra + MLflow)
- Educational notebooks (Kaggle/Colab)

## What This Repo Does NOT Contain
- Adapters for specific robot hardware
- Fine-tuned models trained on proprietary data
- Production safety configurations
- Any IP belonging to specific commercial projects

## Architecture
This project uses a 3-layer separation:
1. **Public** (this repo): algorithms, envs, training, eval — Apache-2.0
2. **Template** (`robotics-platform-template`): HAL interfaces — Apache-2.0
3. **Private** (local only): hardware adapters, proprietary configs — All Rights Reserved

The private layer is never pushed to any public repository.

## License
Apache-2.0. See [LICENSE](LICENSE).
Copyright 2026 Arthur Mouraud.
