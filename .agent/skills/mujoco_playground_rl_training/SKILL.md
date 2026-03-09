---
description: How to set up and train RL locomotion policies for servo robots using MuJoCo Playground on Windows + WSL2
---

# MuJoCo Playground RL Training for Servo Robots (OP3)

This skill documents the setup, pitfalls, and solutions for training RL walking policies for servo-driven bipedal robots (specifically Robotis OP3) using Google DeepMind's `mujoco_playground` framework.

## Context

**OpenServoSim** is a project focused on servo-motor driven bipedal robots. Unlike BLDC joint motors, servos have dead zones, slow feedback, and limited torque. The OP3 (Robotis) is our reference platform.

The hand-tuned control approach (CPG-based walking) hit a fundamental limit: OP3's actuators (Dynamixel XM430, modeled as kp=21.1-40, forcerange=±5-12N) are too weak for a 3.1kg robot to perform large joint motions without falling. **RL is the proven approach** — DeepMind uses it for OP3 Soccer.

---

## Architecture Overview

```
Windows 11 (host)
├── OpenServoSim repo:  C:\GitHub\OpenServoSim
├── MuJoCo viewer:      runs natively or via WSLg
└── WSL2 Ubuntu-24.04
    ├── Venv:           /home/op3_rl_venv/
    ├── JAX + MJX:      GPU-accelerated physics
    ├── mujoco_playground (pip: "playground")
    └── Brax PPO:       RL training algorithm
```

---

## Critical Pitfalls & Solutions

### 1. JAX CUDA Does NOT Work on Native Windows
- JAX only ships CPU wheels for Windows
- **You MUST use WSL2** for GPU-accelerated training
- WSLg (Windows 11) allows MuJoCo viewer GUI to render from WSL2 to Windows desktop — no VcXsrv needed

### 2. RTX 5090 (Blackwell) + JAX CUDA 12 = Segfault
- As of 2026-03, RTX 5090 driver (591.86) reports **CUDA 13.1**
- JAX 0.9.1 ships with **CUDA 12** plugins (`jax-cuda12-plugin`, `jax-cuda12-pjrt`)
- These plugins **segfault at `import jax`** — even with `JAX_PLATFORMS=cpu` set!
- The CUDA plugin auto-loads during import and crashes before any user code runs

**Solution**: Uninstall ALL CUDA plugins:
```bash
pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt \
  nvidia-nvjitlink-cu12 nvidia-nccl-cu12 nvidia-cuda-runtime-cu12 \
  nvidia-cuda-nvrtc-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-cupti-cu12 \
  nvidia-cuda-cccl-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 \
  nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-nvshmem-cu12
```

**Future fix**: Wait for JAX to release CUDA 13 support, then reinstall `jax[cuda13]`.

### 3. mujoco_playground PyPI Name
- The pip package is **`playground`**, NOT `mujoco-playground`
- `pip install playground` → installs as `mujoco_playground` in Python
- Import: `from mujoco_playground import registry`

### 4. Brax PPO + MJX State Incompatibility
- Brax's `ppo.train()` wraps envs in `brax.envs.wrappers.training` 
- These wrappers access `state.pipeline_state` (Brax v2 convention)
- mujoco_playground's `State` (flax `@struct.dataclass`) has `.data` not `.pipeline_state`
- A simple Python wrapper class **won't work** because `jax.vmap(env.reset)` requires JAX-compatible pytree outputs

**Recommended solution** (not yet implemented):
Write a **custom PPO training loop** that directly calls `env.reset()` / `env.step()` without Brax's training wrappers. This avoids the `pipeline_state` incompatibility entirely and gives more control.

Alternative: Monkey-patch the installed `mjx_env.py` to add `pipeline_state` as a property on the State dataclass.

### 5. Menagerie Assets Auto-Download
- mujoco_playground auto-clones `mujoco_menagerie` on first environment load
- Assets go to: `<site-packages>/mujoco_playground/external_deps/mujoco_menagerie/`
- This works automatically — no manual setup needed
- But requires `git` installed in WSL2

### 6. OP3 Joint Sign Conventions
- Zero pose (`ctrl=0` for all actuators) = straight-legged standing at ~279mm height
- Left and right legs have **mirrored joint axes** — same pitch/roll direction requires **opposite sign** for left vs right
- Bending knees even slightly (>0.1 rad) causes face-plant due to weak actuators
- Only safe motions from zero-pose: gentle hip roll sway (±0.06 rad), small arm swing
- Any serious locomotion requires RL — hand-tuning cannot overcome actuator limits

---

## WSL2 Setup Steps

```bash
# 1. Create venv
python3 -m venv /home/op3_rl_venv

# 2. Activate
source /home/op3_rl_venv/bin/activate

# 3. Install (CPU-only, safe for all GPUs)
pip install jax jaxlib brax ml-collections mujoco mujoco-mjx playground

# 4. For GPU with CUDA 12 compatible cards (NOT RTX 5090 as of 2026-03):
# pip install 'jax[cuda12]' instead of jax jaxlib

# 5. Verify
python -c "import jax; print(jax.default_backend())"
# Should print: cpu (or gpu if cuda works)

python -c "from mujoco_playground import registry; env = registry.load('Op3Joystick'); print(f'OK, action_size={env.action_size}')"
# Should print: OK, action_size=20
```

---

## Op3Joystick Environment Details

| Parameter | Value |
|-----------|-------|
| Environment name | `Op3Joystick` |
| Action size | 20 (all joint actuators) |
| Observation size | 147 (49 features × 3 history frames) |
| Observation | gyro(3) + gravity(3) + command(3) + qpos_delta(20) + last_act(20) |
| Control dt | 0.02s (50 Hz) |
| Sim dt | 0.004s (250 Hz) |
| Action scale | 0.3 rad (offset from default_pose) |
| Init keyframe | `stand_bent_knees` |
| Termination | torso_z < 0.21m OR gravity_z < 0.85 OR joint limits |
| PPO envs (GPU) | 8192 |
| PPO timesteps | 100,000,000 |

### Reward Function

| Component | Weight | Purpose |
|-----------|--------|---------|
| tracking_lin_vel | +1.5 | Follow velocity command (core) |
| tracking_ang_vel | +0.8 | Follow turn command |
| orientation | -5.0 | Stay upright |
| lin_vel_z | -2.0 | No bouncing |
| torques | -0.0002 | Energy efficiency |
| action_rate | -0.01 | Smooth motions |
| feet_slip | -0.1 | No sliding feet |
| termination | -1.0 | Don't fall |
| energy | -0.0001 | Power saving |

---

## File Locations in OpenServoSim

```
training/
├── train_op3_walk.py     # PPO training script (WIP: needs custom training loop)
├── setup_wsl.sh          # WSL2 helper script
└── checkpoints/          # saved policy checkpoints

examples/
├── 01_hello_mujoco.py    # Load model + viewer
├── 02_breathing.py       # Gentle sway animation
├── 03_simple_walk.py     # State machine + balance feedback (limited)
└── 04_rl_inference.py    # Load RL policy + WASD keyboard control (WIP)

models/reference/robotis_op3/
├── scene.xml             # Original scene (weak actuators)
├── scene_enhanced.xml    # Enhanced actuators (kp=40, ±12N)
└── op3.xml               # Robot model (21 joints, 20 actuators)
```

---

## Next Steps (for future AI)

1. **Fix GPU training**: Either wait for JAX CUDA 13 support, or write a custom CUDA 12-compatible setup with driver downgrade
2. **Write custom PPO loop**: Bypass Brax's training wrappers, directly use `env.reset()`/`env.step()` with a hand-written PPO in JAX
3. **Train policy**: 100M steps on GPU should take ~10-30 min
4. **Export & deploy**: Save params → load in `04_rl_inference.py` → MuJoCo viewer with WASD control
5. **Sim-to-real**: Map trained policy to real Dynamixel servo commands via OpenServoSim's hardware interface
