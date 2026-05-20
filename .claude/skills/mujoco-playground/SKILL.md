# Skill: mujoco-playground

Use this skill when creating or modifying MuJoCo Playground environments.

## Custom Env Registration

```python
from mujoco_playground import register
from mujoco_playground._src.manipulation.panda_pick_cube import PandaPickCube
import jax.numpy as jnp

class CubeReachV1(PandaPickCube):
    """Dense reach-only reward, 200-step episodes."""

    def compute_reward(self, obs, info):
        dist = jnp.linalg.norm(obs["ee_pos"] - obs["cube_pos"])
        reach_reward = 1.0 - jnp.tanh(5.0 * dist)
        success = dist < 0.05
        return reach_reward + 5.0 * success.astype(jnp.float32)

register("CubeReachV1", CubeReachV1)
```

## Vectorized Rollouts (T4 friendly)
```python
import jax
env = make("CubeReachV1")
reset_fn = jax.vmap(env.reset)
step_fn = jax.vmap(env.step)
```

## Warp Backend Setup
mujoco_playground 0.2.0 uses Warp by default. No extra config needed on T4.
For CPU fallback: set env var `MUJOCO_PLAYGROUND_BACKEND=numpy` (slow, testing only).

## Rules
- Keep custom envs as minimal subclasses (override reward + config, not physics)
- Always register with a versioned name (`CubeReachV1`, not `CubeReach`)
- Domain randomization: keep minimal for free-tier GPU budget
