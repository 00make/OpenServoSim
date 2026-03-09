"""
OpenServoSim - Serial Bus Servo Communication Driver

Implements the LX-15D serial bus servo protocol for real hardware deployment.
Handles half-duplex UART communication, broadcast synchronization, and
position feedback reading.

Protocol frame format:
    [0x55][0x55][ID][Length][CMD][Data...][Checksum]
    Checksum = ~(ID + Length + CMD + Data) & 0xFF

Key commands:
    SERVO_MOVE_TIME_WRITE       (0x01): Move to position with timing
    SERVO_MOVE_TIME_WAIT_WRITE  (0x07): Write position, wait for sync
    SERVO_MOVE_START            (0x0B): Broadcast sync trigger (ID=0xFE)
    SERVO_POS_READ              (0x1C): Read current position
    SERVO_ID_READ               (0x0E): Read servo ID

Broadcast synchronization workflow:
    1. Send SERVO_MOVE_TIME_WAIT_WRITE to each servo individually
    2. Send SERVO_MOVE_START with broadcast ID (0xFE)
    3. All servos move simultaneously

NOTE: This module requires `pyserial` for real hardware.
      Install with: pip install pyserial
"""

import struct
import time
from dataclasses import dataclass, field

# Try to import serial, but don't fail if not available (sim-only mode)
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


# --- Protocol Constants ---
FRAME_HEADER = bytes([0x55, 0x55])
BROADCAST_ID = 0xFE

# Commands
CMD_SERVO_MOVE_TIME_WRITE = 0x01
CMD_SERVO_MOVE_TIME_READ = 0x02
CMD_SERVO_MOVE_TIME_WAIT_WRITE = 0x07
CMD_SERVO_MOVE_START = 0x0B
CMD_SERVO_MOVE_STOP = 0x0C
CMD_SERVO_ID_WRITE = 0x0D
CMD_SERVO_ID_READ = 0x0E
CMD_SERVO_POS_READ = 0x1C
CMD_SERVO_OR_MOTOR_MODE_WRITE = 0x1D
CMD_SERVO_LOAD_OR_UNLOAD_WRITE = 0x1F


@dataclass
class ServoState:
    """Current state of a single servo."""
    id: int
    target_position: float = 0.0  # degrees
    current_position: float = 0.0  # degrees
    offset: float = 0.0  # calibration offset in degrees
    is_loaded: bool = True


class ServoBus:
    """
    Serial bus servo communication driver.
    
    Manages communication with multiple servos on a single half-duplex
    UART bus. Supports both individual and broadcast commands.
    
    Usage:
        bus = ServoBus(port="COM3", baudrate=115200)
        bus.add_servo(0, offset=-2.5)
        bus.add_servo(1, offset=1.0)
        ...
        
        # Set targets individually, then sync
        bus.set_target(0, 90.0)
        bus.set_target(1, 45.0)
        bus.sync_move(duration_ms=100)
        
        # Read positions
        pos = bus.read_position(0)
    """

    # LX-15D angle conversion: position = angle / 0.24
    ANGLE_TO_POS = 1.0 / 0.24
    POS_TO_ANGLE = 0.24

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        timeout: float = 0.01,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.servos: dict[int, ServoState] = {}
        self._serial = None

    def connect(self) -> bool:
        """Open serial port connection."""
        if not HAS_SERIAL:
            print("[ServoBus] pyserial not installed. Install with: pip install pyserial")
            return False

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            print(f"[ServoBus] Connected to {self.port} at {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"[ServoBus] Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Close serial port connection."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            print("[ServoBus] Disconnected")

    def add_servo(self, servo_id: int, offset: float = 0.0) -> None:
        """
        Register a servo on the bus.
        
        Args:
            servo_id: Servo ID (0-253)
            offset: Calibration offset in degrees
        """
        self.servos[servo_id] = ServoState(id=servo_id, offset=offset)

    def set_target(self, servo_id: int, angle_deg: float) -> None:
        """
        Set target position for a servo (does NOT move yet).
        
        Use sync_move() after setting all targets to trigger simultaneous movement.
        
        Args:
            servo_id: Servo ID
            angle_deg: Target angle in degrees
        """
        if servo_id in self.servos:
            self.servos[servo_id].target_position = angle_deg

    def sync_move(self, duration_ms: int = 100) -> None:
        """
        Execute broadcast synchronization:
        1. Send SERVO_MOVE_TIME_WAIT_WRITE to each servo
        2. Send SERVO_MOVE_START broadcast to trigger all simultaneously
        
        Args:
            duration_ms: Movement duration in milliseconds
        """
        if not self._serial:
            return

        # Step 1: Write targets to all servos (wait mode)
        for servo_id, state in self.servos.items():
            if not state.is_loaded:
                continue
            position = int(
                (state.target_position + state.offset) * self.ANGLE_TO_POS
            )
            position = max(0, min(1000, position))  # LX-15D range: 0-1000

            data = struct.pack("<HH", position, duration_ms)
            self._send_command(servo_id, CMD_SERVO_MOVE_TIME_WAIT_WRITE, data)

        # Step 2: Broadcast sync trigger
        self._send_command(BROADCAST_ID, CMD_SERVO_MOVE_START, b"")

    def read_position(self, servo_id: int) -> float | None:
        """
        Read current position from a servo.
        
        Note: Half-duplex UART requires a ~2ms delay after writing
        before reading the response.
        
        Args:
            servo_id: Servo ID to read from
            
        Returns:
            Current position in degrees, or None if read failed
        """
        if not self._serial:
            return None

        self._send_command(servo_id, CMD_SERVO_POS_READ, b"")
        time.sleep(0.002)  # Half-duplex switching delay

        response = self._read_response()
        if response and len(response) >= 2:
            position = struct.unpack("<H", response[:2])[0]
            angle = position * self.POS_TO_ANGLE
            if servo_id in self.servos:
                self.servos[servo_id].current_position = angle
            return angle

        return None

    def unload_all(self) -> None:
        """Disable torque on all servos (safe shutdown)."""
        for servo_id in self.servos:
            self._send_command(
                servo_id,
                CMD_SERVO_LOAD_OR_UNLOAD_WRITE,
                bytes([0]),
            )
            self.servos[servo_id].is_loaded = False

    def _send_command(self, servo_id: int, cmd: int, data: bytes) -> None:
        """Build and send a protocol frame."""
        if not self._serial:
            return

        length = len(data) + 3  # ID + Length + CMD + data
        frame = bytearray(FRAME_HEADER)
        frame.append(servo_id)
        frame.append(length)
        frame.append(cmd)
        frame.extend(data)

        # Checksum = ~(ID + Length + CMD + Data) & 0xFF
        checksum = servo_id + length + cmd + sum(data)
        checksum = (~checksum) & 0xFF
        frame.append(checksum)

        self._serial.write(frame)
        self._serial.flush()

    def _read_response(self) -> bytes | None:
        """Read and parse a response frame from the bus."""
        if not self._serial:
            return None

        # Look for header
        header = self._serial.read(2)
        if header != FRAME_HEADER:
            return None

        # Read ID and length
        id_len = self._serial.read(2)
        if len(id_len) < 2:
            return None

        servo_id, length = id_len
        remaining = self._serial.read(length - 1)  # CMD + data + checksum

        if len(remaining) < length - 1:
            return None

        # Verify checksum
        expected = (~(servo_id + length + sum(remaining[:-1]))) & 0xFF
        if remaining[-1] != expected:
            return None

        # Return data portion (skip CMD byte, exclude checksum)
        return bytes(remaining[1:-1])
