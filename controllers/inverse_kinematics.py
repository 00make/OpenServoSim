"""
OpenServoSim - Servo Leg Inverse Kinematics

Python implementation of the footCont() geometric IK solver.
Maps Cartesian foot positions (x, y, h) to 5-DOF joint angles
for servo-based bipedal legs.

Leg kinematic chain:
    K0  (Hip Pitch)   ← 40mm offset from torso
    K1  (Hip Roll)
    H   (Knee)        ← thigh_length (65mm)
    A0  (Ankle Pitch) ← shin_length (65mm)
    A1  (Ankle Roll)  ← 24.5mm offset to foot

Key engineering approximation:
    Since thigh and shin are equal length, knee angle ≈ 2 × bend angle.
    This is not exact linkage IK, but sufficient for small-to-medium angles
    and drastically simplifies computation (critical for microcontrollers).

Reference:
    servo_robot_knowledge.md, Section IV
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class LegDimensions:
    """Physical dimensions of the leg in meters."""

    hip_offset: float = 0.040    # K1 to K0 distance (40mm)
    thigh_length: float = 0.065  # K0 to H distance (65mm)
    shin_length: float = 0.065   # H to A0 distance (65mm)
    ankle_offset: float = 0.0245 # A0 to A1 distance (24.5mm)

    @property
    def total_leg(self) -> float:
        """Total leg length from hip to ankle."""
        return self.hip_offset + self.thigh_length + self.shin_length + self.ankle_offset

    @property
    def max_extension(self) -> float:
        """Maximum K0-to-A0 straight-line distance."""
        return self.thigh_length + self.shin_length  # 130mm

    @property
    def joint_offset(self) -> float:
        """Combined offset from K1-K0 and A0-A1 distances."""
        return self.hip_offset + self.ankle_offset  # 64.5mm


class ServoLegIK:
    """
    Geometric inverse kinematics for a 5-DOF servo leg.

    Solves for joint angles given desired foot position relative to hip.

    Joint order (output array):
        [0] hip_pitch  (K0)
        [1] hip_roll   (K1)
        [2] knee       (H)
        [3] ankle_pitch (A0)
        [4] ankle_roll  (A1)
    """

    def __init__(self, dims: LegDimensions | None = None):
        self.dims = dims or LegDimensions()
        self._max_reach = self.dims.max_extension * 0.995  # Safety margin

    def solve(
        self,
        x: float = 0.0,
        y: float = 0.0,
        h: float = 0.160,
    ) -> np.ndarray:
        """
        Compute joint angles for desired foot position.

        Args:
            x: Forward/backward distance in meters (forward = positive)
            y: Lateral distance in meters (outward = positive)
            h: Vertical distance from hip to ground in meters

        Returns:
            Array of 5 joint angles in radians:
            [hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll]
        """
        # Step 1: Compute K0-to-A0 straight-line distance
        # Account for the joint offsets (hip_offset + ankle_offset = 64.5mm)
        lateral_dist = np.sqrt(y**2 + h**2) - self.dims.joint_offset
        k = np.sqrt(x**2 + lateral_dist**2)

        # Clamp to maximum extension
        k = min(k, self._max_reach)

        # Ensure minimum extension to avoid singularity
        k = max(k, 0.01)

        # Step 2: Swing angle (forward/backward lean)
        swing = np.arcsin(np.clip(x / k, -1.0, 1.0))

        # Step 3: Bend angle (from cosine rule, simplified for equal-length links)
        cos_arg = k / self.dims.max_extension
        cos_arg = np.clip(cos_arg, -1.0, 1.0)
        bend = np.arccos(cos_arg)

        # Step 4: Joint angles (simplified model for equal thigh/shin)
        hip_pitch = bend + swing           # K0 = bend + swing
        knee = 2.0 * bend                  # H = 2 × bend
        ankle_pitch = bend - swing         # A0 = bend - swing

        # Step 5: Roll angles (lateral balance)
        hip_roll = np.arctan2(y, h)        # K1
        ankle_roll = -hip_roll             # A1 = -K1 (keep foot parallel)

        return np.array([hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll])

    def forward(self, joint_angles: np.ndarray) -> dict:
        """
        Forward kinematics: compute foot position from joint angles.
        Useful for verification and visualization.

        Args:
            joint_angles: [hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll]

        Returns:
            dict with 'x', 'y', 'h' foot position in meters
        """
        hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll = joint_angles

        # Reconstruct swing and bend from joint angles
        bend = knee / 2.0
        swing = hip_pitch - bend

        # Reconstruct K0-to-A0 distance
        k = self.dims.max_extension * np.cos(bend)

        # Reconstruct x (forward/backward)
        x = k * np.sin(swing)

        # Reconstruct lateral distance
        lateral_dist = k * np.cos(swing) if np.abs(swing) < np.pi / 2 else 0.01

        # Add back joint offsets
        total_dist = lateral_dist + self.dims.joint_offset

        # Decompose into y and h using roll angle
        h = total_dist * np.cos(hip_roll)
        y = total_dist * np.sin(hip_roll)

        return {"x": float(x), "y": float(y), "h": float(h)}
