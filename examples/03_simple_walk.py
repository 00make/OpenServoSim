"""
=============================================================================
  OpenServoSim - Milestone 3: Simple Walk
=============================================================================

  OP3 performs a basic alternating gait — shifting weight side to side
  and stepping forward. This demonstrates CPG-based walking with
  servo-compatible 50Hz position control.

  Run:
      python examples/03_simple_walk.py

  What you'll see:
      - OP3 starts in a standing pose
      - Shifts weight to the right leg
      - Left leg lifts and steps forward
      - Shifts weight to left leg
      - Right leg lifts and steps forward
      - Repeat for several cycles

  Key concepts:
      - CPG (Central Pattern Generator) using sine waves
      - Weight shift via hip_roll
      - Foot lifting via knee + ankle coordination
      - Phase-locked arm swing for balance
=============================================================================
"""

import os
import time
import numpy as np
import mujoco
import mujoco.viewer


# ---------------------------------------------------------------------------
#  Gait Parameters
# ---------------------------------------------------------------------------
GAIT_FREQ = 0.8          # Hz — one full stride (2 steps) cycle
STEP_HEIGHT = 0.025       # meters — how high to lift feet
STRIDE_LENGTH = 0.02      # meters — forward distance per step
WEIGHT_SHIFT = 0.12       # radians — hip roll for weight transfer
SQUAT_DEPTH = 0.3         # radians — base knee bend for stability
ARM_SWING = 0.4           # radians — counter-rotation arm swing

# Actuator map
ACT_MAP = {}


def build_actuator_map(model):
    global ACT_MAP
    ACT_MAP = {}
    for i in range(model.nu):
        ACT_MAP[model.actuator(i).name] = i


def ctrl(data, name, value):
    if name in ACT_MAP:
        data.ctrl[ACT_MAP[name]] = value


def get_model_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    enhanced = os.path.join(
        project_root, "models", "reference", "robotis_op3", "scene_enhanced.xml"
    )
    if os.path.exists(enhanced):
        return enhanced
    return os.path.join(
        project_root, "models", "reference", "robotis_op3", "scene.xml"
    )


def get_body_height(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    return data.xpos[bid][2] if bid >= 0 else 0.0


def cpg_walk(data, t):
    """
    Generate a simple CPG-based walking gait.

    The gait uses sine/cosine waves with phase offsets:
    - cos(phase) controls lateral weight shift (hip_roll)
    - if phase in [0, pi]: left foot is swinging
    - if phase in [pi, 2*pi]: right foot is swinging

    Foot trajectories are generated with:
    - Hip pitch:  forward/backward swing
    - Knee + Ankle: foot lifting (only during swing phase)
    """
    phase = 2 * np.pi * GAIT_FREQ * t

    # --- Lateral weight shift ---
    # cos(phase): +1 = weight on left, -1 = weight on right
    lateral = WEIGHT_SHIFT * np.cos(phase)

    # Left hip_roll axis is [-1,0,0], positive = leg inward
    # Right hip_roll axis is [-1,0,0], positive = leg inward
    ctrl(data, "l_hip_roll_act", -lateral)
    ctrl(data, "r_hip_roll_act", -lateral)
    ctrl(data, "l_ank_roll_act",  lateral)
    ctrl(data, "r_ank_roll_act",  lateral)

    # --- Swing phase detection ---
    # sin(phase) positive = left leg swing, negative = right leg swing
    swing_signal = np.sin(phase)

    # Smooth step function: only lift foot during swing
    left_swing = max(0, swing_signal)     # 0 to 1 when left swings
    right_swing = max(0, -swing_signal)   # 0 to 1 when right swings

    # --- Forward/backward hip pitch ---
    # Left hip pitch axis [0,1,0]: positive = forward
    # Right hip pitch axis [0,-1,0]: positive = backward, so negate
    forward = STRIDE_LENGTH * 10  # Convert to approximate radian scale

    l_hip_p = SQUAT_DEPTH * 0.5 + forward * np.sin(phase)
    r_hip_p = -(SQUAT_DEPTH * 0.5 + forward * np.sin(phase + np.pi))

    ctrl(data, "l_hip_pitch_act", l_hip_p)
    ctrl(data, "r_hip_pitch_act", r_hip_p)

    # --- Knee bend (base squat + swing lift) ---
    l_knee = SQUAT_DEPTH + left_swing * 0.4   # Extra bend when swinging
    r_knee = -(SQUAT_DEPTH + right_swing * 0.4)

    ctrl(data, "l_knee_act", l_knee)
    ctrl(data, "r_knee_act", r_knee)

    # --- Ankle pitch compensation ---
    l_ank_p = SQUAT_DEPTH * 0.5 - forward * np.sin(phase) * 0.5
    r_ank_p = -(SQUAT_DEPTH * 0.5 - forward * np.sin(phase + np.pi) * 0.5)

    ctrl(data, "l_ank_pitch_act", l_ank_p)
    ctrl(data, "r_ank_pitch_act", r_ank_p)

    # --- Counter-rotation arm swing ---
    arm_phase = np.sin(phase)
    ctrl(data, "l_sho_pitch_act",  0.3 + ARM_SWING * arm_phase)
    ctrl(data, "r_sho_pitch_act", -(0.3 - ARM_SWING * arm_phase))
    ctrl(data, "l_sho_roll_act", -0.3)
    ctrl(data, "r_sho_roll_act",  0.3)

    # --- Static joints ---
    ctrl(data, "l_hip_yaw_act", 0.0)
    ctrl(data, "r_hip_yaw_act", 0.0)
    ctrl(data, "l_el_act", 0.0)
    ctrl(data, "r_el_act", 0.0)
    ctrl(data, "head_pan_act", 0.0)
    ctrl(data, "head_tilt_act", 0.0)

    return phase


def main():
    print("=" * 60)
    print("  OpenServoSim - Milestone 3: Simple Walk")
    print("=" * 60)

    model_path = get_model_path()
    print(f"  Model: {os.path.basename(model_path)}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    build_actuator_map(model)

    # --- Settle into initial squat (2 seconds) ---
    print("\n  Settling into standing squat pose...")
    for i in range(model.nu):
        data.ctrl[i] = 0.0

    # Set base squat
    ctrl(data, "l_hip_pitch_act", SQUAT_DEPTH * 0.5)
    ctrl(data, "l_knee_act", SQUAT_DEPTH)
    ctrl(data, "l_ank_pitch_act", SQUAT_DEPTH * 0.5)
    ctrl(data, "r_hip_pitch_act", -SQUAT_DEPTH * 0.5)
    ctrl(data, "r_knee_act", -SQUAT_DEPTH)
    ctrl(data, "r_ank_pitch_act", -SQUAT_DEPTH * 0.5)
    ctrl(data, "l_sho_pitch_act",  0.3)
    ctrl(data, "r_sho_pitch_act", -0.3)
    ctrl(data, "l_sho_roll_act", -0.3)
    ctrl(data, "r_sho_roll_act",  0.3)

    for _ in range(3000):
        mujoco.mj_step(model, data)

    h0 = get_body_height(model, data)
    print(f"  Starting height: {h0*1000:.1f}mm")

    # --- Walking loop ---
    print("\n  Starting CPG walking gait...")
    print(f"  Frequency: {GAIT_FREQ} Hz  |  Squat: {np.degrees(SQUAT_DEPTH):.0f} deg")
    print("  " + "-" * 56)
    print(f"  {'Time':>6s}  {'Height':>8s}  {'X-pos':>8s}  {'Steps':>6s}  {'Status':>8s}")
    print("  " + "-" * 56)

    control_dt = 1.0 / 50.0  # 50Hz
    steps_per_ctrl = int(control_dt / model.opt.timestep)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_time = 0.0
        last_print = -1.0
        step_count = 0

        while viewer.is_running():
            loop_start = time.time()

            # CPG control update
            phase = cpg_walk(data, sim_time)

            # Physics sub-steps
            for _ in range(steps_per_ctrl):
                mujoco.mj_step(model, data)

            sim_time = data.time

            # Count steps (full phase wraps)
            new_steps = int(sim_time * GAIT_FREQ * 2)
            step_count = new_steps

            # Print every 2 seconds
            if sim_time - last_print >= 2.0:
                last_print = sim_time
                h = get_body_height(model, data)
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
                x_pos = data.xpos[bid][0]
                status = "Walking" if h > 0.12 else "DOWN"
                print(
                    f"  {sim_time:6.1f}s  "
                    f"{h*1000:7.1f}mm  "
                    f"{x_pos*1000:+7.1f}mm  "
                    f"{step_count:6d}  "
                    f"{status:>8s}"
                )

            viewer.sync()

            elapsed = time.time() - loop_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f"\n  Walk complete! {sim_time:.1f}s, {step_count} steps")


if __name__ == "__main__":
    main()
