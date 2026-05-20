# Skill: kaggle-deployment

Use this skill for Kaggle notebook training setup (2×T4, 9h sessions).

## Hardware Check
```python
import torch
assert torch.cuda.device_count() >= 1, "No GPU found"
print(f"GPUs: {torch.cuda.device_count()}, CUDA: {torch.version.cuda}")
```

## Checkpoint Resume Pattern (CRITICAL for 9h limit)
```python
from pathlib import Path

CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoints = sorted(CHECKPOINT_DIR.glob("step_*.ckpt"))
if checkpoints:
    latest = checkpoints[-1]
    state = torch.load(latest)
    start_step = state["step"]
    print(f"Resuming from step {start_step}")
else:
    start_step = 0
    print("Starting fresh")
```

## Kaggle Secrets for Tokens
```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
hf_token = secrets.get_secret("HF_TOKEN")
wandb_key = secrets.get_secret("WANDB_API_KEY")  # optional
```

## Install via uv
```bash
!pip install uv
!uv pip install lerobot==0.5.1 mujoco==3.5.0 mujoco_playground==0.2.0 hydra-core==1.3.2 mlflow==3.12.0
```

## GPU Skip Marker
```python
import pytest
@pytest.mark.gpu
def test_training_run():
    ...  # skip in CPU-only CI with: pytest -m "not gpu"
```

## Rules
- Always add checkpoint resume logic before any training loop
- Save checkpoints at least every 5000 steps (< 30 min on T4)
- Push to Hub ONLY at end of session (not during training)
- Use `private=True` for intermediate checkpoints, `private=False` for released models
