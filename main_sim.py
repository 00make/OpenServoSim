"""
OpenServoSim - Simulation Entry Point

Run a MuJoCo simulation of the servo biped robot with optional
servo physics modeling and controller selection.

Usage:
    python main_sim.py                    # Headless breathing demo
    python main_sim.py --render           # With visualization
    python main_sim.py --controller uvc   # Use UVC controller
    python main_sim.py --no-servo-model   # Skip servo physics layer

Examples:
    # Quick test: robot does sinusoidal knee bending ("breathing")
    python main_sim.py --render

    # UVC balance test: push the robot and watch it recover
    python main_sim.py --controller uvc --render --duration 10
"""

import argparse
import time
import numpy as np
import os

from sim.mujoco_env import MuJoCoServoEnv
from sim.servo_model import ServoModel, ServoConfig


def create_controller(name: str):
    """Factory function to create a controller by name."""
    if name == "breathing":
        return None  # Use built-in breathing demo
    elif name == "uvc":
        from controllers.uvc_controller import UVCController
        return UVCController(control_freq=50.0)
    else:
        raise ValueError(f"Unknown controller: {name}")


def breathing_demo(t: float, num_actuators: int) -> np.ndarray:
    """
    Simple sinusoidal "breathing" motion for testing.
    
    Makes the robot's knees bend in a slow sine wave,
    verifying that the simulation pipeline works end-to-end.
    
    Args:
        t: Current time in seconds
        num_actuators: Number of actuators
    
    Returns:
        Target positions for all actuators
    """
    targets = np.zeros(num_actuators)
    
    # Sinusoidal knee bending (0.5 Hz, amplitude ±0.3 rad)
    knee_angle = 0.5 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
    
    # Symmetric for both legs
    # Joint order: hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll
    for leg_offset in [0, 5]:  # Right leg (0-4), Left leg (5-9)
        targets[leg_offset + 0] = knee_angle / 2     # Hip pitch
        targets[leg_offset + 1] = 0.0                 # Hip roll
        targets[leg_offset + 2] = knee_angle           # Knee
        targets[leg_offset + 3] = knee_angle / 2     # Ankle pitch
        targets[leg_offset + 4] = 0.0                 # Ankle roll
    
    return targets


def main():
    parser = argparse.ArgumentParser(
        description="OpenServoSim - Servo Biped Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--render", action="store_true", help="Enable real-time visualization"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="breathing",
        choices=["breathing", "uvc"],
        help="Controller to use (default: breathing demo)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Simulation duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--control-freq",
        type=float,
        default=50.0,
        help="Control loop frequency in Hz (default: 50)",
    )
    parser.add_argument(
        "--no-servo-model",
        action="store_true",
        help="Disable servo physics simulation (ideal actuators)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to MJCF model file (default: models/servo_biped/servo_biped.xml)",
    )
    args = parser.parse_args()

    # Resolve model path
    if args.model is None:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models", "servo_biped", "servo_biped.xml",
        )
    else:
        model_path = args.model

    print("=" * 60)
    print("  OpenServoSim - Servo Biped Simulation")
    print("=" * 60)
    print(f"  Model:      {os.path.basename(model_path)}")
    print(f"  Controller: {args.controller}")
    print(f"  Duration:   {args.duration}s")
    print(f"  Frequency:  {args.control_freq} Hz")
    print(f"  Servo Model: {'OFF' if args.no_servo_model else 'ON'}")
    print(f"  Render:     {'ON' if args.render else 'OFF'}")
    print("=" * 60)

    # --- Initialize environment ---
    sim_timestep = 0.002  # 2ms = 500Hz physics
    env = MuJoCoServoEnv(model_path, timestep=sim_timestep)
    env.reset()

    print(f"\n  Model loaded: {env.model.nq} qpos, {env.num_actuators} actuators")

    # --- Initialize servo model (optional) ---
    servo_model = None
    if not args.no_servo_model:
        servo_model = ServoModel(
            num_servos=env.num_actuators,
            control_freq=args.control_freq,
            sim_timestep=sim_timestep,
        )
        servo_model.reset()

    # --- Initialize controller ---
    controller = create_controller(args.controller)
    if controller is not None:
        controller.reset()

    # --- Main simulation loop ---
    control_period = 1.0 / args.control_freq
    total_steps = int(args.duration / sim_timestep)
    steps_per_control = int(control_period / sim_timestep)

    print(f"\n  Running {total_steps} simulation steps...")
    print(f"  Control update every {steps_per_control} steps ({control_period*1000:.1f}ms)")
    print()

    sim_time = 0.0
    last_control_time = -control_period  # Force first update

    for step in range(total_steps):
        sim_time = step * sim_timestep

        # --- Control update (at control frequency) ---
        if sim_time - last_control_time >= control_period:
            last_control_time = sim_time

            if controller is None:
                # Breathing demo
                targets = breathing_demo(sim_time, env.num_actuators)
            else:
                # Use controller
                imu_data = env.get_imu_data()
                joint_pos = env.get_joint_positions()
                joint_vel = env.get_joint_velocities()
                targets = controller.compute(imu_data, joint_pos, joint_vel)

            # Apply servo model (delay, dead zone, filter)
            if servo_model is not None:
                targets = servo_model.apply(targets, sim_time)

            env.set_actuator_targets(targets)

        # --- Physics step ---
        env.step()

        # --- Render (if enabled) ---
        if args.render:
            env.render()
            # Throttle to real-time
            time.sleep(max(0, sim_timestep - 0.0001))

        # --- Periodic info ---
        if step > 0 and step % (steps_per_control * 10) == 0:
            imu = env.get_imu_data()
            height = env.get_torso_height()
            contacts = env.get_foot_contacts()
            print(
                f"  t={sim_time:5.2f}s | "
                f"height={height*1000:.1f}mm | "
                f"pitch={np.degrees(imu['pitch']):+5.1f}° | "
                f"roll={np.degrees(imu['roll']):+5.1f}° | "
                f"feet: R={'▓' if contacts['right'] else '░'} "
                f"L={'▓' if contacts['left'] else '░'}"
            )

    print(f"\n  Simulation complete ({args.duration}s)")

    # --- Cleanup ---
    env.close()


if __name__ == "__main__":
    main()
