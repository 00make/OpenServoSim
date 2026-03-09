"""
=============================================================================
  OpenServoSim — Milestone 1: Hello MuJoCo
=============================================================================

  这是 OpenServoSim 教程系列的第一步：
  加载 Robotis OP3 模型并在 MuJoCo 中可视化。

  运行:
      python examples/01_hello_mujoco.py

  你将看到:
      - MuJoCo 3D 窗口弹出
      - OP3 机器人站在地板上
      - 可以用鼠标旋转/缩放视角
      - 控制台打印所有关节和执行器信息

  操作提示:
      - 左键拖拽: 旋转视角
      - 右键拖拽: 平移
      - 滚轮: 缩放
      - 按 ESC 或关闭窗口退出
=============================================================================
"""

import os
import time
import numpy as np
import mujoco
import mujoco.viewer


def get_model_path():
    """Resolve the path to the OP3 scene.xml model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(
        project_root, "models", "reference", "robotis_op3", "scene.xml"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Make sure you have the OP3 reference model downloaded."
        )
    return model_path


def print_model_info(model):
    """Print detailed information about the loaded model."""
    print("\n" + "=" * 60)
    print("  📋 OP3 模型信息 (Model Info)")
    print("=" * 60)
    print(f"  总自由度 (DOF):      {model.nv}")
    print(f"  关节数 (Joints):     {model.njnt}")
    print(f"  执行器数 (Actuators): {model.nu}")
    print(f"  刚体数 (Bodies):     {model.nbody}")
    print(f"  仿真步长 (Timestep): {model.opt.timestep}s")

    # Print all joints
    print("\n  🔩 关节列表 (Joints):")
    print("  " + "-" * 56)
    print(f"  {'编号':>4s}  {'名称':<25s}  {'类型':>6s}  {'范围':>12s}")
    print("  " + "-" * 56)
    for i in range(model.njnt):
        name = model.joint(i).name
        jnt_type = ["free", "ball", "slide", "hinge"][model.jnt_type[i]]
        if model.jnt_limited[i]:
            lo, hi = model.jnt_range[i]
            range_str = f"[{np.degrees(lo):+.0f}°, {np.degrees(hi):+.0f}°]"
        else:
            range_str = "unlimited"
        print(f"  {i:4d}  {name:<25s}  {jnt_type:>6s}  {range_str:>12s}")

    # Print all actuators
    print(f"\n  ⚙️  执行器列表 (Actuators — 全部为 position 位置伺服):")
    print("  " + "-" * 56)
    print(f"  {'编号':>4s}  {'名称':<25s}  {'增益 Kp':>8s}  {'力矩限制':>10s}")
    print("  " + "-" * 56)
    for i in range(model.nu):
        name = model.actuator(i).name
        kp = model.actuator_gainprm[i][0]
        fmin, fmax = model.actuator_forcerange[i]
        print(f"  {i:4d}  {name:<25s}  {kp:8.1f}  [{fmin:+.1f}, {fmax:+.1f}]N")

    print()


def main():
    print("=" * 60)
    print("  🤖 OpenServoSim — Milestone 1: Hello MuJoCo")
    print("=" * 60)
    print(f"  MuJoCo version: {mujoco.__version__}")

    # --- Load model ---
    model_path = get_model_path()
    print(f"\n  Loading: {os.path.basename(model_path)}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Print model info
    print_model_info(model)

    # --- Set initial standing pose ---
    # Give a slight knee bend so the robot starts in a stable squat
    # instead of falling from a fully extended position
    print("  🦵 设置初始站立姿态 (Setting initial standing pose)...")
    initial_pose = {
        # Slight knee bend for stability
        "l_hip_pitch": 0.3,
        "l_knee": -0.6,
        "l_ank_pitch": 0.3,
        "r_hip_pitch": -0.3,
        "r_knee": 0.6,
        "r_ank_pitch": -0.3,
    }

    for joint_name, angle in initial_pose.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            # Set the actuator control for position servos
            act_name = f"{joint_name}_act"
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if act_id >= 0:
                data.ctrl[act_id] = angle

    # Let the simulation settle for a moment
    print("  ⏳ 等待物理稳定 (Settling physics)...")
    for _ in range(500):
        mujoco.mj_step(model, data)

    # --- Launch interactive viewer ---
    print("\n  🖥️  启动 MuJoCo 交互式窗口...")
    print("  ┌─────────────────────────────────────────┐")
    print("  │  鼠标左键拖拽 = 旋转    滚轮 = 缩放     │")
    print("  │  鼠标右键拖拽 = 平移    ESC = 退出       │")
    print("  └─────────────────────────────────────────┘")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()

            # Step physics
            mujoco.mj_step(model, data)

            # Keep control active (position actuators hold pose)
            for joint_name, angle in initial_pose.items():
                act_name = f"{joint_name}_act"
                act_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
                )
                if act_id >= 0:
                    data.ctrl[act_id] = angle

            # Sync viewer
            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    total_time = time.time() - start_time
    print(f"\n  ✅ 仿真结束！运行时间: {total_time:.1f}s")
    print("  下一步: python examples/02_breathing.py")


if __name__ == "__main__":
    main()
