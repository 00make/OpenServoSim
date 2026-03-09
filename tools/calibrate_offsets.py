"""
OpenServoSim - Servo Zero-Position Calibration Tool

Interactive tool for calibrating servo offsets. Each servo has a
mechanical mounting tolerance that needs to be compensated with
a software offset value.

Usage:
    python tools/calibrate_offsets.py --port COM3

Workflow:
    1. Connect to servo bus
    2. For each servo: move to 0° and ask user to verify alignment
    3. User adjusts offset until joint is at true geometric zero
    4. Save calibration values to a JSON file
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Run the interactive calibration procedure."""
    print("=" * 50)
    print("OpenServoSim - Servo Calibration Tool")
    print("=" * 50)
    print()
    print("This tool helps you calibrate servo zero positions.")
    print("Connect your servo bus and run with --port <PORT>.")
    print()
    print("TODO: Implement interactive calibration loop.")
    print("  1. Move each servo to estimated 0°")
    print("  2. User adjusts with +/- keys")
    print("  3. Save offsets to calibration.json")
    print()

    # Default calibration template
    calibration = {
        "servo_offsets": {
            "r_hip_pitch": 0.0,
            "r_hip_roll": 0.0,
            "r_knee": 0.0,
            "r_ankle_pitch": 0.0,
            "r_ankle_roll": 0.0,
            "l_hip_pitch": 0.0,
            "l_hip_roll": 0.0,
            "l_knee": 0.0,
            "l_ankle_pitch": 0.0,
            "l_ankle_roll": 0.0,
        },
        "unit": "degrees",
        "description": "Servo zero-position calibration offsets",
    }

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "calibration.json",
    )
    with open(output_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"Template saved to: {output_path}")


if __name__ == "__main__":
    main()
