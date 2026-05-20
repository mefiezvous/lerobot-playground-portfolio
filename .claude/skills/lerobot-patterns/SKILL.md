# Skill: lerobot-patterns

Use this skill when working with LeRobot datasets, policies, or HuggingFace Hub interactions.

## LeRobotDataset v3.0 Patterns

### Create a dataset
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset.create(
    repo_id="mefiezvous/cube-reach-v1-dataset",
    fps=20,
    features={...},
)
```

### Push to Hub
```python
dataset.push_to_hub(tags=["robotics", "mujoco"])
```

### Load from Hub
```python
dataset = LeRobotDataset("mefiezvous/cube-reach-v1-dataset")
```

## Policy Loading Patterns

### ACT
```python
from lerobot.common.policies.act.modeling_act import ACTPolicy
policy = ACTPolicy.from_pretrained("mefiezvous/cube-reach-v1-act")
```

### DiffusionPolicy
```python
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
policy = DiffusionPolicy.from_pretrained("mefiezvous/cube-reach-v1-diffusion")
```

## Rules
- Always use v3.0 format (Parquet + MP4), not v2
- Always set `private=False` when pushing benchmark models
- Always log dataset repo_id and commit hash to MLflow
- Never push private datasets to public Hub
