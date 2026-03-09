"""
Systematic joint axis analysis for OP3 model.
Reads every joint + actuator from MuJoCo model and prints axis, range, default.
Then applies +0.5 rad to each arm joint one at a time to see physical effect.
"""
import os, sys
import mujoco
import numpy as np

def get_model_path():
    p = r"c:\GitHub\OpenServoSim\models\reference\robotis_op3\scene_enhanced.xml"
    return p if os.path.exists(p) else r"c:\GitHub\OpenServoSim\models\reference\robotis_op3\scene.xml"

model = mujoco.MjModel.from_xml_path(get_model_path())
data = mujoco.MjData(model)

print("=" * 80)
print("  OP3 Joint & Actuator Axis Map")
print("=" * 80)

# Print ALL joints
print(f"\n{'Joint Name':<25s} {'Axis (x,y,z)':<15s} {'Range (deg)':<20s} {'Type'}")
print("-" * 75)
for i in range(model.njnt):
    jnt = model.joint(i)
    if jnt.type == 0:  # free joint
        print(f"  {jnt.name:<23s} {'FREE':<15s}")
        continue
    axis = jnt.axis
    rng = np.degrees(model.jnt_range[i]) if model.jnt_limited[i] else [0, 0]
    ax_str = f"({axis[0]:+.0f},{axis[1]:+.0f},{axis[2]:+.0f})"
    rng_str = f"[{rng[0]:+6.1f}, {rng[1]:+6.1f}]" if model.jnt_limited[i] else "unlimited"
    print(f"  {jnt.name:<23s} {ax_str:<15s} {rng_str:<20s}")

# Print ALL actuators and their corresponding joints
print(f"\n{'Actuator Name':<25s} {'Joint':<25s} {'Gear'}")
print("-" * 60)
for i in range(model.nu):
    act = model.actuator(i)
    # Get the joint this actuator controls
    trntype = act.trntype
    if trntype == 0:  # joint
        jnt_id = act.trnid[0]
        jnt_name = model.joint(jnt_id).name
    else:
        jnt_name = "?"
    gear = act.gear[0]
    print(f"  {act.name:<23s} {jnt_name:<25s} {gear:+.1f}")

# Now test arm joints specifically
print(f"\n{'='*80}")
print("  ARM JOINT PHYSICAL EFFECT TEST")
print(f"  For each arm joint, apply +1.0 rad and check body position change")
print(f"{'='*80}\n")

arm_actuators = [n for n in [model.actuator(i).name for i in range(model.nu)] 
                 if 'sho' in n or 'el' in n]

for act_name in arm_actuators:
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
    jnt_id = model.actuator(act_id).trnid[0]
    jnt = model.joint(jnt_id)
    
    # Reset
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    # Get default hand position  
    # Find the end effector body for this arm
    side = "l" if act_name.startswith("l") else "r"
    # Look for the forearm/hand body
    hand_body = f"{side}_el_link"
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand_body)
    pos0 = data.xpos[hid].copy()
    
    # Apply +1.0 rad
    data.ctrl[act_id] = 1.0
    for _ in range(500):
        mujoco.mj_step(model, data)
    pos1 = data.xpos[hid].copy()
    
    delta = pos1 - pos0
    # Describe the physical motion
    max_axis = np.argmax(np.abs(delta))
    axis_names = ["X(forward)", "Y(left)", "Z(up)"]
    direction = "+" if delta[max_axis] > 0 else "-"
    
    print(f"  {act_name:<20s}  axis=({jnt.axis[0]:+.0f},{jnt.axis[1]:+.0f},{jnt.axis[2]:+.0f})")
    print(f"    +1.0 rad -> hand moves: dX={delta[0]*100:+5.1f}cm  dY={delta[1]*100:+5.1f}cm  dZ={delta[2]*100:+5.1f}cm")
    print(f"    Primary motion: {direction}{axis_names[max_axis]}")
    
    # Also test -1.0 rad
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    data.ctrl[act_id] = -1.0
    for _ in range(500):
        mujoco.mj_step(model, data)
    pos2 = data.xpos[hid].copy()
    delta2 = pos2 - pos0
    max_axis2 = np.argmax(np.abs(delta2))
    direction2 = "+" if delta2[max_axis2] > 0 else "-"
    print(f"    -1.0 rad -> hand moves: dX={delta2[0]*100:+5.1f}cm  dY={delta2[1]*100:+5.1f}cm  dZ={delta2[2]*100:+5.1f}cm")
    print(f"    Primary motion: {direction2}{axis_names[max_axis2]}")
    print()

# Summary: what values make both arms hang down naturally?
print(f"{'='*80}")
print("  CONCLUSION: Values for natural arm-down pose")
print(f"{'='*80}")
print("  Test: which sho_roll value brings each arm DOWN (negative Z)?")
print()

for side in ["l", "r"]:
    act_name = f"{side}_sho_roll_act"
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
    hand_body = f"{side}_el_link"
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand_body)
    
    for val in [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        data.ctrl[act_id] = val
        for _ in range(1000):
            mujoco.mj_step(model, data)
        pos = data.xpos[hid]
        print(f"  {act_name} = {val:+4.1f}  ->  hand at Z={pos[2]*100:+5.1f}cm  Y={pos[1]*100:+5.1f}cm")
    print()
