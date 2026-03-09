"""
OpenServoSim - Servo Physics Model

Simulates the real-world imperfections of serial bus servos:
- Communication delay (half-duplex UART latency)
- Position dead zone (servo PID internal dead band)
- Response smoothing (first-order low-pass filter)
- Position-only control (no torque passthrough)

This layer sits between the controller and the MuJoCo environment,
transforming "ideal" position commands into "realistic" servo behavior.

Usage:
    servo_model = ServoModel(num_servos=10, control_freq=50.0, sim_timestep=0.002)
    
    # In the control loop:
    realistic_targets = servo_model.apply(ideal_targets, sim_time)
    env.set_actuator_targets(realistic_targets)
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ServoConfig:
    """Configuration for servo physical properties."""

    # Communication delay in seconds (half-duplex UART read/write switching)
    comm_delay: float = 0.002  # 2ms typical for LX-15D

    # Dead zone threshold in radians (~1.9° from UVC dead zone analysis)
    dead_zone: float = 0.033

    # Low-pass filter alpha (0 = no filtering, 1 = complete filtering)
    # Controls how "sluggish" the servo response is
    # alpha = dt / (tau + dt), where tau is the time constant
    filter_alpha: float = 0.3

    # Maximum angular velocity in rad/s (servo speed limit)
    # LX-15D: ~60°/0.16s at 7.4V ≈ 6.5 rad/s
    max_velocity: float = 6.5

    # Position noise standard deviation in radians
    # Simulates servo position jitter
    position_noise: float = 0.005


class ServoModel:
    """
    Servo physics simulation layer.
    
    Wraps ideal position commands with realistic servo imperfections
    to improve Sim-to-Real transfer.
    """

    def __init__(
        self,
        num_servos: int = 10,
        control_freq: float = 50.0,
        sim_timestep: float = 0.002,
        config: ServoConfig | None = None,
    ):
        """
        Initialize the servo model.

        Args:
            num_servos: Number of servo actuators
            control_freq: Control loop frequency in Hz
            sim_timestep: MuJoCo simulation timestep in seconds
            config: Servo configuration (uses defaults if None)
        """
        self.num_servos = num_servos
        self.control_freq = control_freq
        self.control_period = 1.0 / control_freq
        self.sim_timestep = sim_timestep
        self.config = config or ServoConfig()

        # Internal state
        self._current_positions = np.zeros(num_servos)
        self._filtered_targets = np.zeros(num_servos)
        self._last_control_time = 0.0

        # Delay buffer: store (time, target) tuples
        self._delay_buffer: list[tuple[float, np.ndarray]] = []

        # Steps per control period
        self._steps_per_control = max(
            1, int(self.control_period / self.sim_timestep)
        )

    def reset(self) -> None:
        """Reset internal state."""
        self._current_positions = np.zeros(self.num_servos)
        self._filtered_targets = np.zeros(self.num_servos)
        self._last_control_time = 0.0
        self._delay_buffer.clear()

    def apply(
        self, targets: np.ndarray, sim_time: float
    ) -> np.ndarray:
        """
        Apply servo physics to ideal position targets.

        This method simulates:
        1. Communication delay (targets are delayed by comm_delay)
        2. Dead zone (small changes are ignored)
        3. Low-pass filtering (smooth response)
        4. Velocity limiting (max servo speed)
        5. Position noise (jitter)

        Args:
            targets: Ideal position targets in radians, shape (num_servos,)
            sim_time: Current simulation time in seconds

        Returns:
            Realistic position targets after servo physics, shape (num_servos,)
        """
        assert len(targets) == self.num_servos

        # --- Step 1: Communication delay ---
        # Add current targets to delay buffer
        self._delay_buffer.append((sim_time, targets.copy()))

        # Find the most recent target that has "arrived" (past the delay)
        delayed_targets = self._current_positions.copy()
        while (
            self._delay_buffer
            and self._delay_buffer[0][0] <= sim_time - self.config.comm_delay
        ):
            _, delayed_targets = self._delay_buffer.pop(0)

        # --- Step 2: Dead zone ---
        delta = delayed_targets - self._filtered_targets
        mask = np.abs(delta) > self.config.dead_zone
        active_targets = np.where(mask, delayed_targets, self._filtered_targets)

        # --- Step 3: Low-pass filter ---
        alpha = self.config.filter_alpha
        self._filtered_targets = (
            alpha * self._filtered_targets + (1.0 - alpha) * active_targets
        )

        # --- Step 4: Velocity limiting ---
        max_delta = self.config.max_velocity * self.sim_timestep
        position_delta = self._filtered_targets - self._current_positions
        position_delta = np.clip(position_delta, -max_delta, max_delta)
        self._current_positions += position_delta

        # --- Step 5: Position noise ---
        if self.config.position_noise > 0:
            noise = np.random.normal(
                0, self.config.position_noise, self.num_servos
            )
            return self._current_positions + noise

        return self._current_positions.copy()

    @property
    def steps_per_control(self) -> int:
        """Number of simulation steps per control update."""
        return self._steps_per_control

    def should_update(self, sim_time: float) -> bool:
        """
        Check if it's time for a new control update.

        Args:
            sim_time: Current simulation time

        Returns:
            True if enough time has passed since last control update
        """
        if sim_time - self._last_control_time >= self.control_period:
            self._last_control_time = sim_time
            return True
        return False
