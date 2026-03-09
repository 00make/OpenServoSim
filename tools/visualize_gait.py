"""
OpenServoSim - Gait Trajectory Visualization Tool

Plots foot trajectories, joint angle profiles, and UVC integration
values over a gait cycle. Useful for debugging and tuning gait parameters.

Usage:
    python tools/visualize_gait.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

from controllers.inverse_kinematics import ServoLegIK


def plot_ik_workspace():
    """Visualize the IK solver's reachable workspace."""
    ik = ServoLegIK()

    # Generate a grid of foot positions
    x_range = np.linspace(-0.04, 0.04, 20)
    h_range = np.linspace(0.10, 0.19, 20)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("OpenServoSim - IK Workspace Analysis", fontsize=14)

    # Subplot 1: Knee angle vs height
    for x in [0.0, 0.02, -0.02]:
        knee_angles = []
        for h in h_range:
            joints = ik.solve(x=x, y=0.0, h=h)
            knee_angles.append(np.degrees(joints[2]))
        axes[0].plot(h_range * 1000, knee_angles, label=f"x={x*1000:.0f}mm")

    axes[0].set_xlabel("Hip Height (mm)")
    axes[0].set_ylabel("Knee Angle (degrees)")
    axes[0].set_title("Knee Angle vs Hip Height")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Hip pitch vs forward reach
    for h in [0.14, 0.16, 0.18]:
        hip_pitch = []
        for x in x_range:
            joints = ik.solve(x=x, y=0.0, h=h)
            hip_pitch.append(np.degrees(joints[0]))
        axes[1].plot(x_range * 1000, hip_pitch, label=f"h={h*1000:.0f}mm")

    axes[1].set_xlabel("Forward Reach (mm)")
    axes[1].set_ylabel("Hip Pitch (degrees)")
    axes[1].set_title("Hip Pitch vs Forward Reach")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: Foot lift trajectory (sinusoidal)
    fh_max = 20  # mm
    fwct_end = 20
    land_f = 5
    land_b = 3
    t = np.arange(0, fwct_end + 1)

    fh = np.zeros_like(t, dtype=float)
    for i, fwct in enumerate(t):
        if fwct > land_f and fwct <= fwct_end - land_b:
            phase = (fwct - land_f) / (fwct_end - land_f - land_b)
            fh[i] = fh_max * np.sin(np.pi * phase)

    axes[2].plot(t, fh, "b-o", markersize=4)
    axes[2].axvspan(0, land_f, alpha=0.1, color="green", label="Landing (start)")
    axes[2].axvspan(fwct_end - land_b, fwct_end, alpha=0.1, color="red", label="Landing (end)")
    axes[2].set_xlabel("Gait Phase (step)")
    axes[2].set_ylabel("Foot Lift Height (mm)")
    axes[2].set_title("Swing Foot Trajectory")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("docs/assets/workspace_analysis.png", dpi=150, bbox_inches="tight")
    print("Saved to docs/assets/workspace_analysis.png")
    plt.show()


if __name__ == "__main__":
    plot_ik_workspace()
