"""
OpenServoSim - MuJoCo Environment Wrapper

Provides a clean interface for loading and interacting with
the MuJoCo simulation of servo-based bipedal robots.
"""

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    raise ImportError(
        "MuJoCo is required. Install with: pip install mujoco"
    )


class MuJoCoServoEnv:
    """
    MuJoCo environment wrapper for servo biped simulation.
    
    This class handles:
    - Model loading and physics stepping
    - IMU data extraction (pitch, roll, yaw from quaternion)
    - Joint state reading
    - Actuator command application
    - Optional real-time rendering
    
    Usage:
        env = MuJoCoServoEnv("models/servo_biped/servo_biped.xml")
        env.reset()
        
        for _ in range(1000):
            imu_data = env.get_imu_data()
            joint_states = env.get_joint_positions()
            env.set_actuator_targets(targets)
            env.step()
            env.render()  # optional
        
        env.close()
    """

    # Joint name mapping (matches MJCF actuator names)
    JOINT_NAMES = [
        "r_hip_pitch", "r_hip_roll", "r_knee", "r_ankle_pitch", "r_ankle_roll",
        "l_hip_pitch", "l_hip_roll", "l_knee", "l_ankle_pitch", "l_ankle_roll",
    ]

    ACTUATOR_NAMES = [
        "r_hip_pitch_servo", "r_hip_roll_servo", "r_knee_servo",
        "r_ankle_pitch_servo", "r_ankle_roll_servo",
        "l_hip_pitch_servo", "l_hip_roll_servo", "l_knee_servo",
        "l_ankle_pitch_servo", "l_ankle_roll_servo",
    ]

    def __init__(self, model_path: str, timestep: float = 0.002):
        """
        Initialize the MuJoCo environment.

        Args:
            model_path: Path to the MJCF XML file
            timestep: Simulation timestep in seconds (default 2ms)
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._timestep = timestep

        # Cache sensor IDs for fast lookup
        self._imu_quat_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat"
        )
        self._imu_gyro_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"
        )

        # Cache joint and actuator IDs
        self._joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.JOINT_NAMES
        ]
        self._actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.ACTUATOR_NAMES
        ]

    @property
    def num_actuators(self) -> int:
        """Number of actuators (servos)."""
        return len(self.ACTUATOR_NAMES)

    @property
    def timestep(self) -> float:
        """Simulation timestep in seconds."""
        return self._timestep

    def reset(self) -> None:
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n_substeps: int = 1) -> None:
        """
        Advance the simulation by n_substeps.

        Args:
            n_substeps: Number of physics steps to take
        """
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    def get_imu_data(self) -> dict:
        """
        Extract IMU data from simulation sensors.

        Returns:
            dict with keys:
                - pitch: forward/backward tilt in radians (forward lean = negative)
                - roll:  left/right tilt in radians (right lean = positive)
                - yaw:   heading in radians
                - gyro:  angular velocity [x, y, z] in rad/s
        """
        # Get quaternion from framequat sensor
        quat_adr = self.model.sensor_adr[self._imu_quat_id]
        quat = self.data.sensordata[quat_adr : quat_adr + 4].copy()

        # Convert quaternion to Euler angles (ZYX convention)
        # quat = [w, x, y, z] in MuJoCo
        w, x, y, z = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Get gyroscope data
        gyro_adr = self.model.sensor_adr[self._imu_gyro_id]
        gyro = self.data.sensordata[gyro_adr : gyro_adr + 3].copy()

        return {
            "pitch": float(pitch),
            "roll": float(roll),
            "yaw": float(yaw),
            "gyro": gyro,
        }

    def get_joint_positions(self) -> np.ndarray:
        """
        Get current joint positions in radians.

        Returns:
            Array of shape (10,) with joint angles in order of JOINT_NAMES
        """
        positions = np.zeros(len(self._joint_ids))
        for i, jid in enumerate(self._joint_ids):
            positions[i] = self.data.qpos[self.model.jnt_qposadr[jid]]
        return positions

    def get_joint_velocities(self) -> np.ndarray:
        """
        Get current joint velocities in rad/s.

        Returns:
            Array of shape (10,) with joint velocities
        """
        velocities = np.zeros(len(self._joint_ids))
        for i, jid in enumerate(self._joint_ids):
            velocities[i] = self.data.qvel[self.model.jnt_dofadr[jid]]
        return velocities

    def set_actuator_targets(self, targets: np.ndarray) -> None:
        """
        Set position targets for all servo actuators.

        Args:
            targets: Array of shape (10,) with target angles in radians
                     Order matches ACTUATOR_NAMES
        """
        assert len(targets) == self.num_actuators, (
            f"Expected {self.num_actuators} targets, got {len(targets)}"
        )
        for i, aid in enumerate(self._actuator_ids):
            self.data.ctrl[aid] = targets[i]

    def get_torso_height(self) -> float:
        """Get the height of the torso center above ground."""
        torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        return float(self.data.xpos[torso_id, 2])

    def get_foot_contacts(self) -> dict:
        """
        Check if feet are in contact with the ground.

        Returns:
            dict with 'right' and 'left' boolean values
        """
        r_front_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "r_foot_front_touch"
        )
        r_back_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "r_foot_back_touch"
        )
        l_front_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "l_foot_front_touch"
        )
        l_back_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "l_foot_back_touch"
        )

        r_contact = (
            self.data.sensordata[self.model.sensor_adr[r_front_id]] > 0.1
            or self.data.sensordata[self.model.sensor_adr[r_back_id]] > 0.1
        )
        l_contact = (
            self.data.sensordata[self.model.sensor_adr[l_front_id]] > 0.1
            or self.data.sensordata[self.model.sensor_adr[l_back_id]] > 0.1
        )

        return {"right": bool(r_contact), "left": bool(l_contact)}

    def render(self) -> None:
        """Launch or update the interactive viewer."""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()

    def close(self) -> None:
        """Close the viewer and clean up."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
