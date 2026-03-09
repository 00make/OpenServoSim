"""
=============================================================================
  OpenServoSim - Milestone 2: Breathing Motion
=============================================================================

  OP3 does sinusoidal knee bending ("breathing") while standing.
  This verifies that all joint mappings are correct and the control
  loop runs at the intended 50Hz frequency.

  Run:
      python examples/02_breathing.py

  What you'll see:
      - OP3 stands on the ground
      - Knees bend rhythmically in a slow sine wave
      - Console prints real-time joint angles and body height
      - Press ESC or close the window to exit

  Key concepts demonstrated:
      - Position actuator control (servo-like)
      - 50Hz control loop (matching real servo bus bandwidth)
      - Correct joint axis conventions for mirrored legs
=============================================================================
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer


# ---------------------------------------------------------------------------
#  OP3 Joint Map — understanding the axis conventions
# ---------------------------------------------------------------------------
#  Left leg axes:  hip_pitch=[0,1,0]  knee=[0,1,0]  ank_pitch=[0,-1,0]
#  Right leg axes: hip_pitch=[0,-1,0] knee=[0,-1,0]  ank_pitch=[0,1,0]
#
#  This means: for a symmetric squat, left and right need OPPOSITE signs.
#  Positive l_hip_pitch = lean forward; Positive r_hip_pitch = lean backward
# ---------------------------------------------------------------------------

# Actuator name to index mapping (built at runtime)
ACT_MAP = {}


def build_actuator_map(model):
    """Build a name -> index map for all actuators."""
    global ACT_MAP
    ACT_MAP = {}
    for i in range(model.nu):
        ACT_MAP[model.actuator(i).name] = i


def set_ctrl(data, name, value):
    """Set an actuator control value by name."""
    if name in ACT_MAP:
        data.ctrl[ACT_MAP[name]] = value


def get_model_path():
    """Resolve the path to the enhanced OP3 scene."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    # Use our enhanced scene with stronger actuators
    enhanced = os.path.join(
        project_root, "models", "reference", "robotis_op3", "scene_enhanced.xml"
    )
    if os.path.exists(enhanced):
        return enhanced
    # Fallback to original
    return os.path.join(
        project_root, "models", "reference", "robotis_op3", "scene.xml"
    )


def standing_pose(data):
    """
    Set a stable standing pose.
    All joints at 0 = fully extended (straight legs).
    The position actuators will hold this pose via PD control.
    """
    # Hold all joints at zero (straight standing)
    for name in ACT_MAP:
        set_ctrl(data, name, 0.0)


def breathing_pose(data, t, amplitude=0.35, freq=0.4):
    """
    Apply sinusoidal knee bending while maintaining balance.

    The key insight: because left and right leg joint axes are MIRRORED,
    the same physical motion requires OPPOSITE signs for left vs right.

    For a symmetric squat:
      - l_hip_pitch  > 0  (flex forward on axis [0,1,0])
      - r_hip_pitch  < 0  (flex forward on axis [0,-1,0])
      - l_knee       > 0  (bend on axis [0,1,0])
      - r_knee       < 0  (bend on axis [0,-1,0])
      - l_ank_pitch  > 0  (compensate on axis [0,-1,0])
      - r_ank_pitch  < 0  (compensate on axis [0,1,0])
    """
    # Sinusoidal oscillation (offset so it stays bent)
    phase = 2 * np.pi * freq * t
    bend = amplitude * (0.5 + 0.5 * np.sin(phase))  # 0 to amplitude

    # Left leg (positive direction for bending)
    set_ctrl(data, "l_hip_pitch_act",  bend * 0.5)
    set_ctrl(data, "l_knee_act",       bend)
    set_ctrl(data, "l_ank_pitch_act",  bend * 0.5)

    # Right leg (negative direction — mirrored axes)
    set_ctrl(data, "r_hip_pitch_act", -bend * 0.5)
    set_ctrl(data, "r_knee_act",      -bend)
    set_ctrl(data, "r_ank_pitch_act", -bend * 0.5)

    # Arms slightly out for balance
    set_ctrl(data, "l_sho_pitch_act",  0.3)
    set_ctrl(data, "r_sho_pitch_act", -0.3)
    set_ctrl(data, "l_sho_roll_act",  -0.2)
    set_ctrl(data, "r_sho_roll_act",   0.2)

    # Head, hips yaw/roll, ankle roll = hold at zero
    for name in ["head_pan_act", "head_tilt_act",
                 "l_hip_yaw_act", "r_hip_yaw_act",
                 "l_hip_roll_act", "r_hip_roll_act",
                 "l_ank_roll_act", "r_ank_roll_act",
                 "l_el_act", "r_el_act"]:
        set_ctrl(data, name, 0.0)

    return bend


def get_body_height(model, data, body_name="body_link"):
    """Get the height (z) of a body."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid >= 0:
        return data.xpos[bid][2]
    return 0.0


def main():
    print("=" * 60)
    print("  OpenServoSim - Milestone 2: Breathing Motion")
    print("=" * 60)

    model_path = get_model_path()
    print(f"  Model: {os.path.basename(model_path)}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    build_actuator_map(model)
    print(f"  Actuators: {model.nu}")
    print(f"  Timestep: {model.opt.timestep}s")

    # --- Phase 1: Settle into standing pose ---
    print("\n  Phase 1: Settling into standing pose...")
    standing_pose(data)
    for _ in range(2000):  # 2000 steps = ~4s at 500Hz physics
        mujoco.mj_step(model, data)

    h0 = get_body_height(model, data)
    print(f"  Standing height: {h0*1000:.1f}mm")

    # --- Phase 2: Launch viewer with breathing ---
    print("\n  Phase 2: Starting breathing motion (0.4 Hz)")
    print("  " + "-" * 50)
    print(f"  {'Time':>6s}  {'Knee':>8s}  {'Height':>10s}  {'Status':>10s}")
    print("  " + "-" * 50)

    control_freq = 50.0  # Hz — matches real servo bandwidth
    control_dt = 1.0 / control_freq
    physics_steps_per_ctrl = int(control_dt / model.opt.timestep)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_time = 0.0
        last_print = -1.0
        step_count = 0

        while viewer.is_running():
            loop_start = time.time()

            # --- Control update at 50Hz ---
            bend = breathing_pose(data, sim_time)

            # --- Physics sub-steps ---
            for _ in range(physics_steps_per_ctrl):
                mujoco.mj_step(model, data)
                step_count += 1

            sim_time = data.time

            # --- Periodic console output ---
            if sim_time - last_print >= 1.0:
                last_print = sim_time
                h = get_body_height(model, data)
                status = "OK" if h > 0.1 else "FALLEN!"
                print(
                    f"  {sim_time:6.1f}s  "
                    f"{np.degrees(bend):+7.1f} deg  "
                    f"{h*1000:8.1f}mm  "
                    f"{status:>10s}"
                )
                if h < 0.05:
                    print("  [!] Robot has fallen. Try adjusting parameters.")

            # Sync viewer
            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - loop_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f"\n  Done! Simulated {sim_time:.1f}s")
    print("  Next: python examples/03_simple_walk.py")


if __name__ == "__main__":
    main()
