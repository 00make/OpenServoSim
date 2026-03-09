"""
=============================================================================
  OpenServoSim - Milestone 4: RL Inference with MuJoCo Viewer
=============================================================================

  Loads a trained RL policy checkpoint and runs it in the MuJoCo viewer.
  Uses keyboard to send velocity commands (joystick).

  Controls:
    W / S  — forward / backward  (lin_vel_x)
    A / D  — strafe left / right (lin_vel_y)
    Q / E  — turn left / right   (ang_vel_yaw)
    SPACE  — stop (zero command)
    ESC    — quit

  Run from WSL2 (with WSLg for viewer):
    source /home/op3_rl_venv/bin/activate
    python /mnt/c/GitHub/OpenServoSim/examples/04_rl_inference.py <checkpoint_path>
=============================================================================
"""

import os
import sys
import time
import pickle

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

from mujoco_playground import registry


ENV_NAME = "Op3Joystick"

# Velocity command settings
LIN_VEL_X_RANGE = [-0.5, 1.2]
LIN_VEL_Y_RANGE = [-0.6, 0.6]
ANG_VEL_RANGE = [-0.5, 0.5]
VEL_STEP = 0.1


def load_checkpoint(ckpt_path):
    """Load trained policy from checkpoint."""
    inference_path = os.path.join(ckpt_path, "inference_fn.pkl")
    if not os.path.exists(inference_path):
        print(f"Error: {inference_path} not found")
        sys.exit(1)

    with open(inference_path, "rb") as f:
        data = pickle.load(f)

    make_inference_fn = data["make_inference_fn"]
    params = data["params"]
    inference_fn = make_inference_fn(params)
    return inference_fn


class KeyboardController:
    """Captures keyboard input for velocity commands."""

    def __init__(self):
        self.cmd = np.zeros(3)  # [lin_vel_x, lin_vel_y, ang_vel_yaw]

    def key_callback(self, keycode):
        """Process key press."""
        if keycode == ord('W') or keycode == ord('w'):
            self.cmd[0] = min(self.cmd[0] + VEL_STEP, LIN_VEL_X_RANGE[1])
        elif keycode == ord('S') or keycode == ord('s'):
            self.cmd[0] = max(self.cmd[0] - VEL_STEP, LIN_VEL_X_RANGE[0])
        elif keycode == ord('A') or keycode == ord('a'):
            self.cmd[1] = min(self.cmd[1] + VEL_STEP, LIN_VEL_Y_RANGE[1])
        elif keycode == ord('D') or keycode == ord('d'):
            self.cmd[1] = max(self.cmd[1] - VEL_STEP, LIN_VEL_Y_RANGE[0])
        elif keycode == ord('Q') or keycode == ord('q'):
            self.cmd[2] = min(self.cmd[2] + VEL_STEP, ANG_VEL_RANGE[1])
        elif keycode == ord('E') or keycode == ord('e'):
            self.cmd[2] = max(self.cmd[2] - VEL_STEP, ANG_VEL_RANGE[0])
        elif keycode == ord(' '):
            self.cmd[:] = 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python 04_rl_inference.py <checkpoint_path>")
        print("  e.g.: python 04_rl_inference.py training/checkpoints/op3_walk/op3_walk_20250310_050000")
        sys.exit(1)

    ckpt_path = sys.argv[1]

    print("=" * 60)
    print("  OpenServoSim - OP3 RL Inference")
    print("=" * 60)

    # Load policy
    print(f"  Loading checkpoint: {ckpt_path}")
    inference_fn = load_checkpoint(ckpt_path)
    print("  Policy loaded!")

    # Load environment for model and config
    print(f"  Loading environment: {ENV_NAME}")
    env = registry.load(ENV_NAME)
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Initialize with standing keyframe
    kf = mj_model.keyframe("stand_bent_knees")
    mj_data.qpos[:] = kf.qpos
    mujoco.mj_forward(mj_model, mj_data)

    # Get default pose for action offset
    default_pose = kf.qpos[7:]
    action_scale = env._config.action_scale
    obs_history_size = env._config.obs_history_size

    # Build observation manually
    obs_history = np.zeros(obs_history_size * 49)
    last_act = np.zeros(env.action_size)

    kb = KeyboardController()
    ctrl_dt = env._config.ctrl_dt
    n_substeps = int(ctrl_dt / mj_model.opt.timestep)

    print(f"\n  Controls:")
    print(f"    W/S = forward/backward  |  A/D = strafe  |  Q/E = turn")
    print(f"    SPACE = stop  |  ESC = quit")
    print(f"  ctrl_dt={ctrl_dt}s, substeps={n_substeps}")
    print()

    rng = jax.random.PRNGKey(0)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        last_print = -2.0

        while viewer.is_running():
            t0 = time.time()

            # Build observation (same structure as Op3Joystick._get_obs)
            # gyro (3) + gravity (3) + command (3) + qpos_delta (20) + last_act (20) = 49
            torso_id = mj_model.body("body_link").id

            # Gyro from sensor
            gyro_id = mj_model.sensor("gyro").id
            gyro_adr = mj_model.sensor_adr[gyro_id]
            gyro = mj_data.sensordata[gyro_adr:gyro_adr+3]

            # Gravity from sensor
            grav_id = mj_model.sensor("upvector").id
            grav_adr = mj_model.sensor_adr[grav_id]
            gravity = mj_data.sensordata[grav_adr:grav_adr+3]

            # Build single obs
            qpos_delta = mj_data.qpos[7:] - default_pose
            obs_single = np.concatenate([
                gyro,           # 3
                gravity,        # 3
                kb.cmd,         # 3
                qpos_delta,     # 20
                last_act,       # 20
            ])  # = 49

            # Rolling history
            obs_history = np.roll(obs_history, 49)
            obs_history[:49] = obs_single

            # Run policy inference
            rng, act_rng = jax.random.split(rng)
            obs_jax = jnp.array(obs_history)
            action, _ = inference_fn(obs_jax, act_rng)
            action_np = np.array(action)

            # Apply action as motor targets
            motor_targets = default_pose + action_np * action_scale
            ctrl_lower = mj_model.actuator_ctrlrange[:, 0]
            ctrl_upper = mj_model.actuator_ctrlrange[:, 1]
            motor_targets = np.clip(motor_targets, ctrl_lower, ctrl_upper)
            mj_data.ctrl[:] = motor_targets

            last_act = action_np

            # Step simulation
            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)

            # Print status
            if mj_data.time - last_print >= 1.5:
                last_print = mj_data.time
                h = mj_data.xpos[torso_id][2]
                x = mj_data.xpos[torso_id][0]
                status = "Walking" if h > 0.20 else "FALLEN"
                print(
                    f"  t={mj_data.time:5.1f}s  "
                    f"h={h*1000:.0f}mm  x={x*1000:+.0f}mm  "
                    f"cmd=[{kb.cmd[0]:+.1f}, {kb.cmd[1]:+.1f}, {kb.cmd[2]:+.1f}]  "
                    f"[{status}]"
                )

            viewer.sync()
            sleep_time = ctrl_dt - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\n  Done!")


if __name__ == "__main__":
    main()
