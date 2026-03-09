"""
OpenServoSim - UVC (Upper Body Vertical Control) Controller

Python implementation of Dr. Guero's UVC algorithm for servo-based
bipedal balance. This algorithm is specifically designed for low-frequency
position-controlled servos, avoiding the complex dynamics calculations
that BLDC motor controllers require.

Core idea:
    Read IMU tilt angle → convert to hip joint displacement correction
    → accumulate (integrate) with damping → distribute to support/swing legs

The algorithm uses "integration + clamping + decay recovery" to maintain
balance with "memory" — the robot gradually finds its equilibrium.

Reference:
    http://ai2001.ifdef.jp/uvc/code_Eng.html
"""

import numpy as np
from controllers.base_controller import BaseController
from controllers.inverse_kinematics import ServoLegIK


class UVCController(BaseController):
    """
    Upper Body Vertical Control for servo bipedal robots.
    
    Implements the 7-stage gait state machine:
        710: Initialize (move to standing pose)
        720: Calibrate IMU offsets
        730: Monitor tilt (wait for perturbation)
        740: UVC main loop (single leg support, balance recovery)
        750: Vibration dampening
        760: Recovery step (swing other leg)
        770: Return to standing
    """

    def __init__(
        self,
        num_joints: int = 10,
        control_freq: float = 50.0,
        leg_height: float = 0.160,  # Standing hip height in meters
        sway_amplitude: float = 0.015,  # Lateral sway in meters
        uvc_gain_roll: float = 0.25,  # Roll response coefficient (0.25~0.85)
        uvc_gain_pitch: float = 0.25,  # Pitch response coefficient
        dead_zone: float = 0.033,  # Dead zone threshold in radians (~1.9°)
    ):
        super().__init__(num_joints, control_freq)

        # Geometry parameters (meters)
        self.leg_height = leg_height  # autoH
        self.sway_amplitude = sway_amplitude  # sw (lateral sway distance)

        # UVC tuning parameters
        self.uvc_gain_roll = uvc_gain_roll
        self.uvc_gain_pitch = uvc_gain_pitch
        self.dead_zone = dead_zone

        # IK solver
        self.ik = ServoLegIK()

        # --- Internal state ---
        self._mode = 710
        self._mode_counter = 0

        # UVC integration variables (in meters)
        self._dxi = 0.0   # Forward/backward displacement (support leg)
        self._dyi = 0.0   # Lateral displacement (support leg)
        self._dxis = 0.0  # Forward/backward displacement (swing leg)
        self._dyis = 0.0  # Lateral displacement (swing leg)

        # Current leg height (dynamic)
        self._auto_h = leg_height

        # Support leg: 0 = right, 1 = left
        self._jikuasi = 0

        # Gait phase counter
        self._fwct = 0
        self._fwct_end = 20  # Steps per half-gait cycle

        # Gait timing constants
        self._land_f = 5   # Landing phase duration (start)
        self._land_b = 3   # Landing phase duration (end)

        # Foot lift height
        self._fh_max = 0.02  # 20mm max foot lift

        # IMU calibration
        self._imu_offset_pitch = 0.0
        self._imu_offset_roll = 0.0
        self._calibration_count = 0
        self._calibration_sum_pitch = 0.0
        self._calibration_sum_roll = 0.0

    def reset(self) -> None:
        """Reset controller to initialization state."""
        self._mode = 710
        self._mode_counter = 0
        self._dxi = 0.0
        self._dyi = 0.0
        self._dxis = 0.0
        self._dyis = 0.0
        self._auto_h = self.leg_height
        self._jikuasi = 0
        self._fwct = 0
        self._calibration_count = 0
        self._calibration_sum_pitch = 0.0
        self._calibration_sum_roll = 0.0

    def compute(
        self,
        imu_data: dict,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
    ) -> np.ndarray:
        """
        Compute joint targets using UVC algorithm.

        Returns target positions for all 10 joints:
            [r_hip_pitch, r_hip_roll, r_knee, r_ankle_pitch, r_ankle_roll,
             l_hip_pitch, l_hip_roll, l_knee, l_ankle_pitch, l_ankle_roll]
        """
        pitch = imu_data["pitch"] - self._imu_offset_pitch
        roll = imu_data["roll"] - self._imu_offset_roll

        if self._mode == 710:
            return self._mode_init(pitch, roll)
        elif self._mode == 720:
            return self._mode_calibrate(pitch, roll, imu_data)
        elif self._mode == 730:
            return self._mode_monitor(pitch, roll)
        elif self._mode == 740:
            return self._mode_uvc_main(pitch, roll)
        elif self._mode >= 750:
            return self._mode_recovery(pitch, roll)

        return self._compute_standing_pose()

    def _mode_init(self, pitch: float, roll: float) -> np.ndarray:
        """Mode 710: Slowly move to initial standing posture."""
        self._mode_counter += 1
        if self._mode_counter >= 50:  # ~1 second at 50Hz
            self._mode = 720
            self._mode_counter = 0
        return self._compute_standing_pose()

    def _mode_calibrate(
        self, pitch: float, roll: float, imu_data: dict
    ) -> np.ndarray:
        """Mode 720: Calibrate IMU offsets by averaging 100 readings."""
        self._calibration_sum_pitch += imu_data["pitch"]
        self._calibration_sum_roll += imu_data["roll"]
        self._calibration_count += 1

        if self._calibration_count >= 100:
            self._imu_offset_pitch = self._calibration_sum_pitch / 100.0
            self._imu_offset_roll = self._calibration_sum_roll / 100.0
            self._mode = 730
            self._mode_counter = 0

        return self._compute_standing_pose()

    def _mode_monitor(self, pitch: float, roll: float) -> np.ndarray:
        """Mode 730: Monitor tilt, trigger UVC when threshold exceeded."""
        tilt_magnitude = np.sqrt(pitch**2 + roll**2)

        if tilt_magnitude > self.dead_zone:
            # Choose support leg based on roll direction
            if roll > 0:
                self._jikuasi = 0  # Right foot supports (leaning right)
            else:
                self._jikuasi = 1  # Left foot supports (leaning left)

            self._mode = 740
            self._fwct = 0

        return self._compute_standing_pose()

    def _mode_uvc_main(self, pitch: float, roll: float) -> np.ndarray:
        """
        Mode 740: UVC main control loop.
        
        Core algorithm translated from C:
        1. Apply dead zone to combined tilt
        2. Scale by response gain
        3. During single-support phase: integrate tilt into displacement
        4. Compute foot positions via IK
        """
        # --- UVC dead zone processing ---
        k = np.sqrt(pitch**2 + roll**2)
        if k > self.dead_zone:
            k1 = (k - self.dead_zone) / k
            pitch *= k1
            roll *= k1
        else:
            pitch = 0.0
            roll = 0.0

        # --- Apply gain coefficients ---
        rollt = self.uvc_gain_roll * roll
        if self._jikuasi == 0:
            rollt = -rollt  # Flip sign for right support
        pitcht = self.uvc_gain_pitch * pitch

        # --- Single support phase: integrate UVC corrections ---
        if self._fwct > self._land_f and self._fwct <= self._fwct_end - self._land_b:
            # Roll correction (lateral)
            k = np.arctan2(
                self._dyi - self.sway_amplitude, self._auto_h
            )
            kl = self._auto_h / np.cos(k)
            ks = k + rollt
            self._dyi = kl * np.sin(ks) + self.sway_amplitude
            self._auto_h = kl * np.cos(ks)

            # Pitch correction (fore-aft)
            k = np.arctan2(self._dxi, self._auto_h)
            kl = self._auto_h / np.cos(k)
            ks = k + pitcht
            self._dxi = kl * np.sin(ks)
            self._auto_h = kl * np.cos(ks)

            # Clamp integration values (meters)
            self._dyi = np.clip(self._dyi, 0, 0.045)
            self._dxi = np.clip(self._dxi, -0.045, 0.045)

            # Swing leg follows support leg
            self._dyis = self._dyi
            self._dxis = -self._dxi

            # Anti-crossing constraint
            if self._jikuasi == 0:
                kr = -self.sway_amplitude + self._dyi
                kl_val = self.sway_amplitude + self._dyis
            else:
                kl_val = -self.sway_amplitude + self._dyi
                kr = self.sway_amplitude + self._dyis
            if kr + kl_val < 0:
                self._dyis -= kr + kl_val

        # --- Sub-UVC: decay during landing phase ---
        self._uvc_sub()

        # --- Increment gait counter ---
        self._fwct += 1
        if self._fwct > self._fwct_end:
            self._fwct = 0
            self._mode = 750
            self._mode_counter = 0

        # --- Compute foot positions and run IK ---
        return self._compute_gait_pose()

    def _uvc_sub(self) -> None:
        """
        UVC auxiliary: smooth recovery during landing phases.
        
        - During landing: gradually zero out integration values
        - Continuously: restore leg height with exponential decay
        """
        if self._fwct <= self._land_f and self._fwct > 0:
            # Lateral: proportional decay
            remaining = self._land_f + 1 - self._fwct
            if remaining > 0:
                k = self._dyi / remaining
                self._dyi -= k
                self._dyis += k

            # Fore-aft: fixed rate decay
            rst_f = 0.002  # 2mm per step
            if self._dxi > rst_f:
                self._dxi -= rst_f
                self._dxis -= rst_f
            elif self._dxi < -rst_f:
                self._dxi += rst_f
                self._dxis += rst_f

        # Restore leg height (7% exponential recovery)
        if self.leg_height > self._auto_h:
            self._auto_h += (self.leg_height - self._auto_h) * 0.07

        # Minimum leg height
        if self._auto_h < self.leg_height * 0.75:
            self._auto_h = self.leg_height * 0.75

    def _mode_recovery(self, pitch: float, roll: float) -> np.ndarray:
        """Modes 750-770: Vibration damping and recovery step."""
        self._mode_counter += 1

        if self._mode == 750 and self._mode_counter >= 30:
            # Switch support leg and do recovery step
            self._jikuasi ^= 1  # Flip support leg
            # Swap integration values
            self._dxi, self._dxis = self._dxis, self._dxi
            self._dyi, self._dyis = self._dyis, self._dyi
            self._mode = 760
            self._mode_counter = 0
            self._fwct = 0
        elif self._mode == 760:
            self._fwct += 1
            self._uvc_sub()
            if self._fwct > self._fwct_end:
                self._mode = 770
                self._mode_counter = 0
        elif self._mode == 770 and self._mode_counter >= 50:
            self._mode = 730  # Return to monitoring
            self._dxi = 0.0
            self._dyi = 0.0
            self._dxis = 0.0
            self._dyis = 0.0
            self._auto_h = self.leg_height

        return self._compute_gait_pose()

    def _compute_standing_pose(self) -> np.ndarray:
        """Compute joint targets for neutral standing posture."""
        # Simple standing: slight knee bend
        knee_angle = 0.3  # ~17° bend
        return np.array([
            knee_angle / 2, 0, knee_angle, knee_angle / 2, 0,  # Right leg
            knee_angle / 2, 0, knee_angle, knee_angle / 2, 0,  # Left leg
        ])

    def _compute_gait_pose(self) -> np.ndarray:
        """Compute joint targets from current UVC state using IK."""
        # Compute foot positions for support and swing legs
        # Support leg
        support_x = self._dxi
        support_y = self._dyi if self._jikuasi == 0 else -self._dyi
        support_joints = self.ik.solve(support_x, support_y, self._auto_h)

        # Swing leg: compute foot lift
        fh = 0.0
        if self._fwct > self._land_f and self._fwct <= self._fwct_end - self._land_b:
            phase = (self._fwct - self._land_f) / (
                self._fwct_end - self._land_f - self._land_b
            )
            fh = self._fh_max * np.sin(np.pi * phase)

        swing_x = self._dxis
        swing_y = self._dyis if self._jikuasi == 0 else -self._dyis
        swing_joints = self.ik.solve(swing_x, swing_y, self._auto_h + fh)

        # Assemble full joint target array
        targets = np.zeros(10)
        if self._jikuasi == 0:
            # Right = support, Left = swing
            targets[0:5] = support_joints
            targets[5:10] = swing_joints
        else:
            # Left = support, Right = swing
            targets[0:5] = swing_joints
            targets[5:10] = support_joints

        return targets

    def get_info(self) -> dict:
        """Return debug information about UVC state."""
        return {
            "mode": self._mode,
            "fwct": self._fwct,
            "jikuasi": self._jikuasi,
            "dxi": self._dxi,
            "dyi": self._dyi,
            "dxis": self._dxis,
            "dyis": self._dyis,
            "auto_h": self._auto_h,
        }
