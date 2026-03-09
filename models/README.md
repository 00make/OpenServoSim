# Models

This directory contains robot model definitions in [MuJoCo MJCF](https://mujoco.readthedocs.io/en/stable/XMLreference.html) format.

## servo_biped/

A simplified 10-DOF bipedal robot model with position-controlled actuators that mimic serial bus servo behavior.

### Key Design Choices

- **Position actuators** instead of motor actuators — servos accept position commands, not torque
- **Realistic dimensions** from actual hardware IK parameters
- **IMU sensor** on torso for pitch/roll feedback
- **Foot contact sensors** for ground contact detection

### Adding Your Own Model

1. Create a new directory under `models/` (e.g., `models/my_robot/`)
2. Place your MJCF XML file and mesh files there
3. Ensure actuators use `position` type (not `motor`) for servo simulation
4. Add IMU sensor sites on the torso body

### Mesh Files

Place STL or OBJ files in the `meshes/` subdirectory. Reference them in your MJCF file:

```xml
<asset>
  <mesh name="torso_mesh" file="meshes/torso.stl" scale="0.001 0.001 0.001"/>
</asset>
```

> **Note**: Scale factor `0.001` converts millimeters (common in CAD) to meters (MuJoCo's unit).
