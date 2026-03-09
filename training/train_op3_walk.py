"""
=============================================================================
  OpenServoSim - Milestone 4: RL Training for OP3 Walking
=============================================================================

  Trains a PPO policy for the OP3 robot using DeepMind's mujoco_playground
  Op3Joystick environment with Brax PPO.
  
  Key: The mujoco_playground State class needs to be wrapped for Brax
  compatibility (Brax expects .pipeline_state, playground uses .data).

  Requirements (WSL2 with CUDA or CPU):
    pip install 'jax[cuda12]' playground brax ml-collections mujoco

  Run:
    # From WSL2:
    source /home/op3_rl_venv/bin/activate
    JAX_PLATFORMS=cpu python /mnt/c/GitHub/OpenServoSim/training/train_op3_walk.py

  Note: CPU training with reduced params (~5M steps, 256 envs).
  For GPU training, JAX CUDA 12 must match your GPU driver's CUDA version.
=============================================================================
"""

import os
import sys
import time
import json
import pickle
from datetime import datetime
from typing import Any

# Force CPU if no GPU available
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training import types
from mujoco_playground import registry
from mujoco_playground._src import mjx_env

# ===== Configuration =====
ENV_NAME = "Op3Joystick"
NUM_TIMESTEPS = 5_000_000  # Reduced for CPU training
NUM_ENVS = 256  # Reduced for CPU
NUM_EVALS = 5
SEED = 42

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "checkpoints", "op3_walk"
)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class BraxWrapper:
    """Wraps mujoco_playground env to be Brax-PPO compatible.
    
    Brax's ppo.train expects:
    - env.reset(rng) -> State with .pipeline_state
    - env.step(state, action) -> State with .pipeline_state  
    - env.observation_size -> int
    - env.action_size -> int
    
    mujoco_playground's State has .data instead of .pipeline_state.
    This wrapper patches State to add pipeline_state as an alias.
    """

    def __init__(self, env):
        self._env = env

    def _patch_state(self, state: mjx_env.State):
        """Add pipeline_state attribute by monkey-patching."""
        # Brax's training wrapper accesses state.pipeline_state
        # We map it to state.data for compatibility
        # Since State is a flax struct.dataclass, we can't easily add attrs
        # Instead, we return a dict-like wrapper
        return BraxState(state)

    def reset(self, rng):
        state = self._env.reset(rng)
        return self._patch_state(state)

    def step(self, state, action):
        # Unwrap back to MJX state
        mjx_state = state._mjx_state
        new_state = self._env.step(mjx_state, action)
        return self._patch_state(new_state)

    @property
    def observation_size(self):
        return self._env.observation_size

    @property
    def action_size(self):
        return self._env.action_size

    @property
    def unwrapped(self):
        return self._env


class BraxState:
    """State wrapper that adds .pipeline_state for Brax compatibility."""

    def __init__(self, mjx_state):
        self._mjx_state = mjx_state

    @property
    def pipeline_state(self):
        return self._mjx_state.data

    @property
    def obs(self):
        return self._mjx_state.obs

    @property
    def reward(self):
        return self._mjx_state.reward

    @property
    def done(self):
        return self._mjx_state.done

    @property
    def metrics(self):
        return self._mjx_state.metrics

    @property
    def info(self):
        return self._mjx_state.info

    def replace(self, **kwargs):
        # Delegate to underlying state
        new_mjx = self._mjx_state.replace(**kwargs)
        return BraxState(new_mjx)

    def tree_replace(self, params):
        new_mjx = self._mjx_state.tree_replace(params)
        return BraxState(new_mjx)


_start_time = 0


def progress_callback(num_steps, metrics):
    """Called periodically during training to report progress."""
    reward = metrics.get("eval/episode_reward", 0)
    r = float(jnp.mean(reward)) if hasattr(reward, "shape") else float(reward)
    elapsed = time.time() - _start_time
    sps = num_steps / elapsed if elapsed > 0 else 0
    print(
        f"  Step {num_steps:>10,}  |  "
        f"reward: {r:>8.2f}  |  "
        f"SPS: {sps:>8.0f}  |  "
        f"time: {elapsed:>6.0f}s ({elapsed/60:.1f}m)"
    )


def main():
    global _start_time

    print("=" * 70)
    print("  OpenServoSim - OP3 RL Training (Brax PPO)")
    print("=" * 70)

    backend = jax.default_backend()
    devices = jax.devices()
    print(f"  JAX backend: {backend}")
    print(f"  Devices: {devices}")
    
    if backend == "gpu":
        num_ts = 100_000_000
        num_envs = 8192
        print("  GPU mode: full training (100M steps, 8192 envs)")
    else:
        num_ts = NUM_TIMESTEPS
        num_envs = NUM_ENVS
        print(f"  CPU mode: reduced training ({num_ts/1e6:.0f}M steps, {num_envs} envs)")
        print("  WARNING: CPU training is ~100x slower than GPU.")

    # Load environment
    print(f"\n  Loading {ENV_NAME}...")
    env = registry.load(ENV_NAME)
    print(f"  Action size: {env.action_size}")
    print(f"  Observation size: {env.observation_size}")
    print(f"  Control dt: {env._config.ctrl_dt}s")
    
    # Use Brax wrapper
    wrapped_env = BraxWrapper(env)

    print(f"\n  PPO Config:")
    print(f"    num_timesteps:  {num_ts:,}")
    print(f"    num_envs:       {num_envs}")
    print(f"    learning_rate:  3e-4")
    
    print(f"\n  Starting training...")
    print(f"  {'━' * 64}")
    _start_time = time.time()

    try:
        make_inference_fn, params, metrics = ppo.train(
            environment=wrapped_env,
            num_timesteps=num_ts,
            num_evals=NUM_EVALS,
            reward_scaling=1.0,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=20,
            num_minibatches=8 if backend == "cpu" else 32,
            num_updates_per_batch=4,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            num_envs=num_envs,
            batch_size=128 if backend == "cpu" else 256,
            max_devices_per_host=1,
            seed=SEED,
            progress_fn=progress_callback,
        )
    except Exception as e:
        print(f"\n  Training error: {e}")
        print(f"\n  This may be a Brax/MJX API compatibility issue.")
        print(f"  Attempting alternative: direct MJX PPO training...")
        raise

    elapsed = time.time() - _start_time
    print(f"  {'━' * 64}")
    print(f"  Training complete in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")

    # Save checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"op3_walk_{timestamp}")
    os.makedirs(ckpt_path, exist_ok=True)

    with open(os.path.join(ckpt_path, "params.pkl"), "wb") as f:
        pickle.dump(params, f)

    with open(os.path.join(ckpt_path, "inference_fn.pkl"), "wb") as f:
        pickle.dump({"make_inference_fn": make_inference_fn, "params": params}, f)

    meta = {
        "env_name": ENV_NAME,
        "training_time_s": elapsed,
        "num_timesteps": num_ts,
        "num_envs": num_envs,
        "timestamp": timestamp,
        "backend": backend,
        "device": str(devices[0]),
    }
    with open(os.path.join(ckpt_path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Checkpoint: {ckpt_path}")
    print(f"\n  Next: Run inference:")
    print(f"    python examples/04_rl_inference.py {ckpt_path}")


if __name__ == "__main__":
    main()
