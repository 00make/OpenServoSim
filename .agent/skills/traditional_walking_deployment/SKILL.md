---
description: How to deploy a traditional (non-RL) walking engine on a humanoid robot in MuJoCo simulation
---

# Traditional Walking Engine Deployment on MuJoCo Humanoid

This skill documents the complete process of porting a production bipedal walking engine (ROBOTIS OP3) to Python + MuJoCo, including **every pitfall encountered and how to avoid them**.

## Overview

**Goal:** Make a simulated humanoid robot walk using a sinusoidal gait generator + geometric IK, without reinforcement learning.

**Architecture:**
```
Sinusoidal Trajectories (wSin) → Cartesian foot positions → IK solver → Joint angles → MuJoCo actuators
```

**Key files in OpenServoSim:**
- `controllers/robotis_walking.py` — Walking engine (wSin + IK + balance)
- `examples/03b_robotis_walk.py` — Viewer + headless diagnostics

---

## Step 1: Understand the MuJoCo Model Joint Axes

> **⚠️ THIS IS THE #1 SOURCE OF BUGS. Do NOT skip this step.**

### The Problem
MuJoCo joint axes are defined per-joint and can be **different between left and right sides**. The same positive control value can produce OPPOSITE physical motions on mirrored limbs. You CANNOT assume symmetric signs.

### Systematic Approach (MANDATORY)
Before writing ANY control code, run a physical axis test script:

```python
# For each actuator, apply +1.0 and -1.0 rad, measure end-effector displacement
for act_name in actuator_names:
    # Reset simulation
    mujoco.mj_resetData(model, data)
    data.ctrl[act_id] = +1.0
    for _ in range(500): mujoco.mj_step(model, data)
    pos_positive = data.xpos[body_id].copy()
    
    # Repeat with -1.0
    # Compare displacements to determine physical meaning
```

### OP3 Joint Axis Reference Table (Verified by Physical Test)

#### Legs
| Joint | Axis | +ctrl effect | Notes |
|-------|------|-------------|-------|
| `l_hip_pitch` | Y+ | Leg forward | |
| `r_hip_pitch` | Y- | Leg forward | ⚠️ Axis MIRRORED from left |
| `l_knee` | Y+ | Leg extends | |
| `r_knee` | Y- | Leg extends | ⚠️ Axis MIRRORED |
| `l_ank_pitch` | Y- | Toe up | ⚠️ REVERSED from hip/knee! |
| `r_ank_pitch` | Y+ | Toe up | ⚠️ REVERSED from hip/knee! |
| `l_hip_roll` | X- | Pelvis tilts | |
| `r_hip_roll` | X- | Pelvis tilts | Same axis as left |
| `l_ank_roll` | X+ | Foot tilts | |
| `r_ank_roll` | X+ | Foot tilts | Same axis as left |

#### Arms
| Joint | Axis | +ctrl effect | "Arm down" value |
|-------|------|-------------|-----------------|
| `l_sho_pitch` | Y+ | Arm forward | |
| `r_sho_pitch` | Y- | Arm backward | ⚠️ MIRRORED |
| `l_sho_roll` | X- | Arm DOWN | **+1.3** |
| `r_sho_roll` | X- | Arm UP | **-1.3** ⚠️ OPPOSITE effect! |
| `l_el` | X+ | Elbow bends | +0.8 |
| `r_el` | X+ | Elbow bends | +0.8 |

### Critical Lesson: Same Axis ≠ Same Physical Effect
`l_sho_roll` and `r_sho_roll` BOTH have axis `(-1,0,0)` but produce **opposite physical motions** because the arms are on opposite sides of the body. The geometry is mirrored even though the axis vector is identical. **You MUST test physically, not reason from axis vectors alone.**

---

## Step 2: Measure Leg Dimensions from the Model

Do NOT use the real robot's spec sheet. MuJoCo models may have different dimensions.

```python
# Read from XML body positions, not documentation
THIGH_LENGTH = 0.11015   # hip_pitch_link → knee_link z-offset
CALF_LENGTH  = 0.110     # knee_link → ank_pitch_link z-offset  
ANKLE_LENGTH = 0.0265    # ank_pitch_link → foot contact z
```

### Pitfall: Using Wrong Leg Lengths
Using ROBOTIS documentation values instead of MuJoCo XML values caused the IK to compute unreachable positions, leading to actuator saturation and immediate collapse.

---

## Step 3: IK Solver Choice

### 3-DOF Planar IK (Recommended for Initial Deployment)
Uses law of cosines for hip_pitch, knee, ankle_pitch in the sagittal plane. More robust than full 6-DOF. Roll/yaw are handled by the trajectory generator directly.

```python
def solve_ik_simple(x, y, z) -> (hip_pitch, knee, ank_pitch):
    d = sqrt(dx² + dz²)  # Distance from hip to ankle
    knee = π - acos((thigh² + calf² - d²) / (2 * thigh * calf))
    hip_pitch = -(atan2(dx, dz) + asin(calf * sin(knee) / d))
    ank_pitch = -(hip_pitch + knee)  # Keep foot flat
```

### Pitfall: Joint-Space vs Cartesian-Space
Our first attempt (`03_simple_walk.py`) used direct joint-angle state machines. This fails because individual joints being set independently produce uncoordinated motion. The ROBOTIS approach computes foot positions in **Cartesian space** first, then solves IK. This naturally produces coordinated multi-joint motion.

---

## Step 4: Standing Pose Before Walking

### Critical: MuJoCo Zero-Pose Convention
In MuJoCo, `ctrl = 0` means the default XML pose (usually straight legs). Unlike real servos where 0 might mean a specific bent position. Set stance offsets near zero:

```python
init_x_offset = -0.005   # Tiny forward offset
init_y_offset = 0.0      # No lateral needed
init_z_offset = 0.005    # Extremely slight knee bend
```

### Pitfall: Large Stance Offsets
Using ROBOTIS default offsets (designed for real servos at specific positions) caused deep knee bends that MuJoCo actuators couldn't hold → immediate collapse.

---

## Step 5: Balance Feedback (Essential)

Without IMU-based balance feedback, the robot accumulates tilt over 3-4 gait cycles and falls.

### Implementation
```python
def apply_balance(angles, pitch, roll, pitch_gain=0.5, roll_gain=0.3):
    pitch_corr = pitch_gain * pitch  # Same sign as pitch
    angles["l_ank_pitch_act"] += pitch_corr
    angles["r_ank_pitch_act"] += pitch_corr
    
    roll_corr = -roll_gain * roll
    angles["l_ank_roll_act"] += roll_corr
    angles["r_ank_roll_act"] += roll_corr
```

### Pitfall: Wrong Balance Feedback Sign
If pitch correction sign is wrong, the balance feedback **amplifies** the topple instead of correcting it. The robot falls in half the time it would without any feedback. **Always test with and without feedback to verify the sign.**

### Pitfall: Measuring Pitch/Roll from Quaternion
```python
# From body quaternion (w, x, y, z):
pitch = arcsin(clip(2*(qw*qy - qz*qx), -1, 1))
roll  = atan2(2*(qw*qx + qy*qz), 1 - 2*(qx² + qy²))
```

---

## Step 6: The Lateral Sway Bug (Hardest to Find)

### The Bug
The robot appeared stable (height stayed above threshold) but was actually tilting 59° sideways. Height-only monitoring missed it because a lateral lean doesn't immediately reduce body center height.

### Root Cause
An `atan2(swap_y, z)` was computed and applied as hip_roll/ankle_roll joint rotations **ON TOP OF** the Cartesian foot endpoint shifts that already contained `swap_y`. This doubled the lateral displacement.

### Fix
The ROBOTIS walking engine's lateral sway works **solely through foot endpoint Y-position offsets** (`swap_y` added to foot position in Cartesian space). It does NOT use hip_roll angular rotations for sway. Remove all `hip_roll = atan2(swap_y, z)` computations from the output.

### Lesson
**Always monitor ALL orientation axes (pitch AND roll AND yaw), not just height.** A robot can be "upright" by height metric while tilted 60° sideways.

---

## Step 7: Diagnostics System (Build This First Next Time)

### WalkDiagnostics Class
Track per-step: height, pitch, roll, X-position, phase, joint angles. Auto-detect falls and analyze cause.

```python
class WalkDiagnostics:
    FALL_THRESHOLD = 0.18
    
    def record(self, sim_time, height, pitch, roll, phase, angles, x_pos):
        # Store sample
        if height < self.FALL_THRESHOLD and self.fall_time is None:
            self._analyze_fall()  # Analyze last 2s of data
    
    def _analyze_fall(self):
        # Check: pitch topple? roll topple? height collapse?
        # Identify which axis failed first
```

### Lesson
Build diagnostics BEFORE tuning. We wasted multiple iterations because we couldn't distinguish between "stable but barely" and "tilting 59° sideways" without monitoring roll.

---

## Tuning Parameter Reference

### Parameters That Work (OP3 in MuJoCo)
| Parameter | Value | Notes |
|-----------|-------|-------|
| `period_time` | 0.600s | Full gait cycle |
| `dsp_ratio` | 0.2 | 20% double-support |
| `z_move_amplitude` | 0.025m | Foot lift height |
| `y_swap_amplitude` | 0.020m | Lateral body sway |
| `z_swap_amplitude` | 0.004m | Vertical bounce |
| `hip_pitch_offset` | 0.0 | No forward lean needed |
| `x_move_amplitude` | 0.025m | Step length |
| `pelvis_offset` | 0.5° | Small roll bias during swing |
| `arm_swing_gain` | 0.5 | Visible arm counter-swing |
| `balance pitch_gain` | 0.5 | Ankle pitch feedback |
| `balance roll_gain` | 0.3 | Ankle roll feedback |

### Parameters That Caused Failures
| Parameter | Bad Value | Effect |
|-----------|-----------|--------|
| `hip_pitch_offset` | 7° | Forward topple in 2 cycles |
| `init_z_offset` | 0.035m | Deep knee → collapse |
| `z_move_amplitude` | 0.060m | Too high lift → unstable |
| `period_time` | 1.0s | Too slow → accumulates error |

---

## Common Failure Modes and Diagnosis

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Immediate collapse | IK unreachable (wrong leg dims) | Measure from XML |
| Falls forward in 2-3 cycles | hip_pitch_offset too large | Set to 0 |
| Falls sideways with good height | Hip_roll sway doubling lateral shift | Remove angular sway |
| Falls faster WITH balance | Balance feedback sign inverted | Reverse pitch_corr sign |
| One arm up, one arm down | sho_roll sign wrong for one side | Physical axis test |
| Robot walks backward | Joint sign convention wrong | Verify with physical test |
| Walking in circles | Left/right asymmetry | Check all mirrored joint signs |

---

## Workflow Summary

```
1. Read ALL joint axes from XML → build axis reference table
2. Physically test each joint (+/-1.0 rad) → verify signs
3. Measure leg dimensions from XML body positions
4. Implement 3-DOF IK with correct leg lengths
5. Build diagnostics (height + pitch + roll monitoring)
6. Get stable standing first (near-zero offsets)
7. Add walking with small amplitudes
8. Add balance feedback (verify sign!)
9. Tune: increase step size, speed, arm posture
```
