"""Pick-and-Place 专家演示数据收集。

让 Panda 机械臂完成"抓起方块 → 搬运 → 放置"的完整操作序列，
同时记录 (observation, action) 对，保存为 BC 训练数据。

核心技术点：
    1. 程序化构建 MuJoCo 场景（避免 <include> meshdir 问题）
    2. 六阶段笛卡尔空间轨迹规划
    3. 阻尼最小二乘 IK（适配 mj_jacBody）
    4. 肌腱驱动夹爪控制（ctrl=0 闭合, ctrl=255 张开）
    5. 仿真数据采集与 .npz 存储

用法:
    python scripts/collect_pick_place.py                    # 默认参数 + viewer
    python scripts/collect_pick_place.py --no-viewer        # 仅采集数据，不显示 viewer
    python scripts/collect_pick_place.py --output results/my_demo.npz  # 自定义输出
    python scripts/collect_pick_place.py --block-pos 0.5 0.1 0.075   # 改变方块位置
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ==============================================================================
# 1. 场景构建（程序化，避免 <include> meshdir 问题）
# ==============================================================================

SCENE_ASSETS_DIR = "/home/wcs/mujoco/menagerie/franka_emika_panda/assets"
PANDA_XML_PATH = "/home/wcs/mujoco/menagerie/franka_emika_panda/panda.xml"


def build_scene_xml():
    """程序化构建完整的场景 XML 字符串。

    为什么不用 <include>？
        panda.xml 中的 meshdir="assets" 是相对路径。
        嵌套 <include> 时，MuJoCo 的 meshdir 解析会出错，导致找不到 STL 网格。
        Day 1 已验证: 用绝对路径直接加载 panda.xml 可避开此问题。

    解决方案:
        读取 panda.xml 原文 → 将 meshdir 替换为绝对路径 →
        在 </worldbody> 之前注入桌子、方块、地面 → from_xml_string 加载。
    """
    # 读取原始 panda.xml
    panda_str = Path(PANDA_XML_PATH).read_text()

    # 修复 meshdir: 相对 → 绝对
    panda_str = panda_str.replace(
        'meshdir="assets"',
        f'meshdir="{SCENE_ASSETS_DIR}"',
    )

    # 构建场景附加元素
    scene_elements = """
    <!-- ===== 地面 ===== -->
    <geom name="floor" size="0 0 0.01" type="plane" rgba="0.5 0.5 0.5 1"/>

    <!-- ===== 桌子 ===== -->
    <!-- 桌面：40cm × 40cm × 2cm，桌体中心在 z=0.04，桌面在 z=0.06 -->
    <body name="table" pos="0.5 0 0.04">
      <geom name="table_top" type="box" size="0.2 0.2 0.02"
            rgba="0.4 0.4 0.4 1"/>
      <!-- 四条桌腿（纯视觉） -->
      <geom name="table_leg1" type="box" size="0.015 0.015 0.03"
            pos="-0.17 -0.17 -0.04" rgba="0.25 0.25 0.25 1"/>
      <geom name="table_leg2" type="box" size="0.015 0.015 0.03"
            pos="-0.17  0.17 -0.04" rgba="0.25 0.25 0.25 1"/>
      <geom name="table_leg3" type="box" size="0.015 0.015 0.03"
            pos=" 0.17 -0.17 -0.04" rgba="0.25 0.25 0.25 1"/>
      <geom name="table_leg4" type="box" size="0.015 0.015 0.03"
            pos=" 0.17  0.17 -0.04" rgba="0.25 0.25 0.25 1"/>
    </body>

    <!-- ===== 红色方块（可抓取，加高加大接触面积） ===== -->
    <!-- 使用 3 个 slide 关节（仅平移，不允许旋转），防止被挤飞 -->
    <!-- 桌面 z=0.06，方块半高 0.025，方块中心 z=0.085 -->
    <body name="block" pos="0.4 0 0.085">
      <joint type="slide" axis="1 0 0" name="block_x" limited="true" range="-0.5 0.5" damping="1"/>
      <joint type="slide" axis="0 1 0" name="block_y" limited="true" range="-0.5 0.5" damping="1"/>
      <joint type="slide" axis="0 0 1" name="block_z" limited="true" range="0.01 1.0" damping="0"/>
      <geom name="block_geom" type="box" size="0.02 0.02 0.025"
            rgba="0.9 0.15 0.15 1" mass="0.04"
            friction="5.0 0.5 0.1"
            solmix="0.2" solref="0.01 1"/>
    </body>

    <!-- ===== 隐形导向墙（防止方块被碰撞弹飞） ===== -->
    <!-- 4 面薄墙围住方块初始位置，形成 6cm×6cm 的\"笼子\" -->
    <!-- 墙高 3cm（半高 1.5cm），方块被举起时会自然脱离 -->
    <geom name="guide_north" type="box" size="0.03 0.003 0.015"
          pos="0.4 0.033 0.09" rgba="0 0 0 0" contype="1" conaffinity="1"/>
    <geom name="guide_south" type="box" size="0.03 0.003 0.015"
          pos="0.4 -0.033 0.09" rgba="0 0 0 0" contype="1" conaffinity="1"/>
    <geom name="guide_east" type="box" size="0.003 0.03 0.015"
          pos="0.433 0 0.09" rgba="0 0 0 0" contype="1" conaffinity="1"/>
    <geom name="guide_west" type="box" size="0.003 0.03 0.015"
          pos="0.367 0 0.09" rgba="0 0 0 0" contype="1" conaffinity="1"/>
    """

    # 注入到 </worldbody> 之前
    panda_str = panda_str.replace("</worldbody>", scene_elements + "\n  </worldbody>")

    return panda_str


def load_scene():
    """加载程序化构建的场景。"""
    xml_str = build_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    print("[scene] 程序化构建场景: Panda + 桌子 + 方块")
    print(f"        nq={model.nq}, nv={model.nv}, nu={model.nu}")
    return model, data


# ==============================================================================
# 2. IK 求解器（适配 body 而非 site）
# ==============================================================================

def solve_ik_body(
    model, data, body_id, target_pos,
    init_q=None, regularization=0.1,
    max_iter=200, tol=1e-4,
):
    """用阻尼最小二乘法求解逆运动学（body 版本）。

    与 Step 1 的 solve_ik 唯一区别:
        - 使用 mj_jacBody(body_id) 而非 mj_jacSite(site_id)
        - 使用 data.xpos[body_id] 而非 data.site_xpos[site_id]

    原因: panda.xml 完整版无 attachment_site，但有 hand body (id=9)。

    Args:
        model, data: MuJoCo 模型和数据
        body_id: 末端 body 的 ID（hand body）
        target_pos: (3,) 目标位置
        init_q: 初始关节角（前 7 维用于 arm）
        regularization: 阻尼系数 λ
        max_iter: 最大迭代次数
        tol: 收敛容差 (m)

    Returns:
        q_arm: (7,) arm 关节解
        success: bool
    """
    if init_q is not None:
        data.qpos[:7] = init_q

    mujoco.mj_forward(model, data)

    for i in range(max_iter):
        current_pos = data.xpos[body_id].copy()
        error = target_pos - current_pos

        if float(np.linalg.norm(error)) < tol:
            return data.qpos[:7].copy(), True

        # 平移雅可比 (3 × nv)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

        # 阻尼最小二乘
        J = jacp  # (3 × 9), 但只前 7 列对应 arm joints
        JJT = J @ J.T + regularization * np.eye(3)
        delta_q_full = J.T @ np.linalg.solve(JJT, error)

        # 只更新 arm 关节 (前 7 维)，不动手指和 block
        data.qpos[:7] += delta_q_full[:7]

        # 钳制 arm 关节到限位
        for j in range(7):
            if model.jnt_limited[j]:
                lo, hi = model.jnt_range[j]
                qpos_adr = model.jnt_qposadr[j]
                data.qpos[qpos_adr] = np.clip(data.qpos[qpos_adr], lo, hi)

        mujoco.mj_forward(model, data)

    err_norm = float(np.linalg.norm(target_pos - data.xpos[body_id]))
    return data.qpos[:7].copy(), err_norm < tol * 10


# ==============================================================================
# 3. 轨迹定义
# ==============================================================================

def define_phases(block_pos, target_offset=np.array([0.0, 0.15, 0.0])):
    """定义 pick-and-place 六个阶段的关键位点。

    Args:
        block_pos: (3,) 方块初始位置 [x, y, z]
        target_offset: 目标位置相对于初始位置的偏移

    Returns:
        phases: list of dict, 每个 dict 包含:
            - name: 阶段名
            - pos: (3,) 目标 hand 位置
            - gripper: float, 夹爪 ctrl 值 (0=闭, 255=开)
            - hold_steps: 在此位点停留的步数
    """
    b = np.asarray(block_pos, dtype=np.float64)
    t = b + np.asarray(target_offset, dtype=np.float64)

    # hand body 到指尖的 Z 偏移约为 -0.014m（指尖在 hand body 下方）
    # 因此 hand body 需要比方块中心高约 1.5cm，指尖才对齐方块中心
    fingertip_offset_z = 0.015

    z_ready = 0.35                  # 预备高度：方块正上方 35cm（避免碰撞）
    z_above = b[2] + 0.08           # 接近高度：方块上方 8cm
    z_grasp = b[2] + fingertip_offset_z  # 抓取高度（指尖对齐方块）
    z_lift  = b[2] + 0.22           # 搬运高度
    z_place = t[2] + fingertip_offset_z # 放置高度
    z_place_above = t[2] + 0.10     # 撤离高度

    phases = [
        # 阶段 0: 预备 —— 先移到方块正上方高处，避免碰撞
        {"name": "ready",      "pos": np.array([b[0], b[1], z_ready]),
         "gripper": 255, "hold_steps": 120},

        # 阶段 1: 接近 —— 从预备高度降到方块上方
        {"name": "approach",   "pos": np.array([b[0], b[1], z_above]),
         "gripper": 255, "hold_steps": 100},

        # 阶段 2: 下降抓取 —— 降到抓取高度
        {"name": "grasp",      "pos": np.array([b[0], b[1], z_grasp]),
         "gripper": 255, "hold_steps": 100},

        # 阶段 2b: 缓慢完全闭合（充分建立稳定接触）
        {"name": "close",      "pos": np.array([b[0], b[1], z_grasp]),
         "gripper": 0,   "hold_steps": 300},

        # 阶段 3: 慢速抬起（防止加速度太大导致方块滑落）
        {"name": "lift",       "pos": np.array([b[0], b[1], z_lift]),
         "gripper": 0,   "hold_steps": 150},

        # 阶段 4: 横向搬运
        {"name": "transport",  "pos": np.array([t[0], t[1], z_lift]),
         "gripper": 0,   "hold_steps": 120},

        # 阶段 5: 下降放置
        {"name": "place",      "pos": np.array([t[0], t[1], z_place]),
         "gripper": 0,   "hold_steps": 80},

        # 阶段 5b: 张开手指（释放方块，充分时间让其落下）
        {"name": "release",    "pos": np.array([t[0], t[1], z_place]),
         "gripper": 255, "hold_steps": 150},

        # 阶段 6: 撤离（方块此时应已落在桌面上）
        {"name": "retreat",    "pos": np.array([t[0], t[1], z_place_above]),
         "gripper": 255, "hold_steps": 80},
    ]
    return phases


# ==============================================================================
# 4. 轨迹预计算与回放
# ==============================================================================

def precompute_joint_trajectory(model, data, body_id, phases, home_q_arm):
    """对每个阶段的位点求解 IK，生成完整关节轨迹。

    返回:
        joint_traj: list of (q_arm(7), gripper_ctrl) 每帧的目标
    """
    joint_traj = []
    prev_q = home_q_arm.copy()

    total_frames = sum(p["hold_steps"] for p in phases)
    ik_success = 0
    ik_total = 0

    for pi, phase in enumerate(phases):
        # 求解此阶段目标位点的 IK
        q_arm, ok = solve_ik_body(
            model, data, body_id, phase["pos"],
            init_q=prev_q, regularization=0.1, max_iter=200, tol=1e-4,
        )
        ik_total += 1
        if ok:
            ik_success += 1

        # 阶段内插值: 从 prev_q 平滑过渡到 q_arm
        for step in range(phase["hold_steps"]):
            alpha = step / phase["hold_steps"]
            # 用正弦缓入缓出让运动更自然
            alpha_smooth = 0.5 - 0.5 * np.cos(alpha * np.pi)
            interp_q = (1 - alpha_smooth) * prev_q + alpha_smooth * q_arm
            joint_traj.append((interp_q.copy(), phase["gripper"]))

        prev_q = q_arm.copy()

    if ik_total > 0:
        print(f"[ik] 位点 IK: {ik_success}/{ik_total} 收敛")

    return joint_traj


def run_and_record(model, data, body_id, joint_traj, record=True):
    """回放关节轨迹、执行仿真、记录观测-动作对。

    混合控制策略:
        - arm 关节 (0-6): 直接设置 data.qpos（瞬时到达 IK 目标，无 PD 延迟）
        - 手指 (7-8): 通过 data.ctrl + PD 控制器驱动（需要接触动力学来抓取）
        - block 的 freejoint: 由 MuJoCo 物理引擎自动更新

    为什么 arm 用 qpos 直设?
        PD 控制器在高频轨迹跟踪中有延迟，可能导致抓取位姿不准确。
        直设 qpos 等价于"完美控制器"，对于演示数据收集是合理的。
    """
    mujoco.mj_resetData(model, data)

    # 重置 Panda 到 home（只设前 9 维，后面是 block freejoint）
    home_q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
    data.qpos[:9] = home_q
    data.ctrl[:7] = home_q[:7]
    data.ctrl[7] = 255  # 手指张开
    mujoco.mj_forward(model, data)

    observations = [] if record else None
    actions = [] if record else None

    # 保存 block 初始 qpos（freejoint 部分）
    block_qpos_start = data.qpos[9:16].copy()

    for frame_idx, (q_arm, gripper_cmd) in enumerate(joint_traj):
        # --- 直设 arm 关节位置（瞬时，不经过 PD） ---
        data.qpos[:7] = q_arm

        # --- 手指通过 ctrl 驱动（需要接触动力学） ---
        data.ctrl[7] = gripper_cmd

        # --- 保持 block freejoint 的合理值（防止数值漂移） ---
        # 如果方块位置异常（穿透桌面等），不做干预，让物理引擎处理

        # --- 记录（step 之前的状态） ---
        if record:
            obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            act = np.zeros(8)
            act[:7] = q_arm
            act[7] = gripper_cmd
            observations.append(obs)
            actions.append(act)

        # --- 仿真一步（手指接触动力学 + block 自由运动） ---
        mujoco.mj_step(model, data)

    if record:
        return np.array(observations), np.array(actions)
    return None, None


def run_viewer_playback(model, data, body_id, joint_traj):
    """在交互式 viewer 中回放轨迹。"""
    mujoco.mj_resetData(model, data)

    home_q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
    data.qpos[:9] = home_q
    data.ctrl[:7] = home_q[:7]
    data.ctrl[7] = 255
    mujoco.mj_forward(model, data)

    total_frames = len(joint_traj)
    print(f"\n[viewer] 启动交互式 viewer")
    print(f"  总帧数: {total_frames}")
    print(f"  预计时长: {total_frames * model.opt.timestep:.1f}s")
    print(f"  控制: 鼠标拖动旋转 | 滚轮缩放 | Ctrl+拖动平移 | Space 暂停 | Esc 退出")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        while viewer.is_running():
            idx = frame % total_frames
            q_arm, gripper_cmd = joint_traj[idx]

            data.ctrl[:7] = q_arm
            data.ctrl[7] = gripper_cmd

            mujoco.mj_step(model, data)
            viewer.sync()
            frame += 1


# ==============================================================================
# 5. 主入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Panda Pick-and-Place 专家演示数据收集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/collect_pick_place.py                        # 默认 + viewer
  python scripts/collect_pick_place.py --no-viewer             # 仅采集 .npz
  python scripts/collect_pick_place.py --block-pos 0.5 -0.05 0.075
  python scripts/collect_pick_place.py --target-offset 0 0.2 0  # 搬运更远
        """,
    )
    parser.add_argument("--no-viewer", action="store_true",
                        help="仅采集数据，不显示 viewer")
    parser.add_argument("--output", "-o", type=str,
                        default="results/pick_place_demo.npz",
                        help="输出 .npz 路径 (默认: results/pick_place_demo.npz)")
    parser.add_argument("--block-pos", nargs=3, type=float,
                        default=[0.4, 0.0, 0.085],
                        metavar=("X", "Y", "Z"),
                        help="方块初始位置 (默认: 0.5 0 0.075)")
    parser.add_argument("--target-offset", nargs=3, type=float,
                        default=[0.0, 0.15, 0.0],
                        metavar=("DX", "DY", "DZ"),
                        help="目标位置相对偏移 (默认: 0 0.15 0, 即右移15cm)")

    args = parser.parse_args()
    block_pos = np.array(args.block_pos, dtype=np.float64)
    target_offset = np.array(args.target_offset, dtype=np.float64)

    # ---- 加载场景 ----
    model, data = load_scene()
    body_id = model.body("hand").id
    block_id = model.body("block").id

    print(f"[info] hand body id={body_id}")
    print(f"[info] block body id={block_id}")

    # Panda home 配置（arm 部分）
    home_q_arm = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

    # ---- 验证可达性 ----
    print(f"\n[check] 验证关键位点可达性...")
    test_positions = [
        ("block", block_pos),
        ("above_block", block_pos + np.array([0, 0, 0.08])),
        ("lift", block_pos + np.array([0, 0, 0.20])),
        ("target_above", block_pos + target_offset + np.array([0, 0, 0.08])),
    ]
    all_reachable = True
    for name, pos in test_positions:
        _, ok = solve_ik_body(model, data, body_id, pos, init_q=home_q_arm,
                              regularization=0.1, max_iter=200, tol=1e-4)
        status = "✅" if ok else "❌"
        if not ok:
            all_reachable = False
        # 计算实际误差
        data.qpos[:7] = home_q_arm
        mujoco.mj_forward(model, data)
        dist = np.linalg.norm(pos - data.xpos[body_id])
        print(f"  {name}: {pos} → {status} (当前 EE 距目标 {dist:.3f}m)")

    if not all_reachable:
        print("\n⚠ 部分位点不可达！请调整方块位置或搬运距离。")
        # 仍然继续，也许加容差后能跑

    # ---- 定义阶段 ----
    phases = define_phases(block_pos, target_offset)
    print(f"\n[plan] 轨迹阶段 ({len(phases)} 个):")
    for p in phases:
        print(f"  {p['name']:12s} → {p['pos']}  gripper={p['gripper']:3.0f}  "
              f"frames={p['hold_steps']}")

    # ---- 预计算关节轨迹 ----
    print(f"\n[precompute] 预计算关节轨迹...")
    t0 = time.time()
    joint_traj = precompute_joint_trajectory(model, data, body_id, phases, home_q_arm)
    elapsed = time.time() - t0
    print(f"  总帧数: {len(joint_traj)}, 耗时: {elapsed:.2f}s")

    # ---- 采集数据：混合控制策略 ----
    print(f"\n[record] 执行轨迹并记录数据...")
    t0 = time.time()

    mujoco.mj_resetData(model, data)
    home_q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
    data.qpos[:9] = home_q
    data.ctrl[:7] = home_q[:7]
    data.ctrl[7] = 255
    mujoco.mj_forward(model, data)

    observations = []
    actions_list = []
    block_traj = []
    current_finger_pos = 0.04  # 初始张开

    for frame_idx, (q_arm, gripper_cmd) in enumerate(joint_traj):
        # ---- arm: 直接设 qpos（完美跟踪 IK） ----
        data.qpos[:7] = q_arm
        data.qvel[:7] = 0.0

        # ---- 手指: 渐进式开合（避免瞬时挤压方块） ----
        # 目标手指位置: 0.0=闭合, 0.04=张开
        target_finger = 0.0 if gripper_cmd < 128 else 0.04
        # 每帧最多移动 0.001（缓慢渐进）
        step_size = 0.001
        if abs(target_finger - current_finger_pos) < step_size:
            current_finger_pos = target_finger
        elif target_finger < current_finger_pos:
            current_finger_pos -= step_size
        else:
            current_finger_pos += step_size

        data.qpos[7] = current_finger_pos
        data.qpos[8] = current_finger_pos

        # 记录（step 之前）
        obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
        act = np.zeros(8)
        act[:7] = q_arm
        act[7] = gripper_cmd
        observations.append(obs)
        actions_list.append(act)

        # 仿真
        mujoco.mj_step(model, data)
        block_traj.append(data.xpos[block_id].copy())

    obs = np.array(observations)
    act = np.array(actions_list)

    elapsed = time.time() - t0
    print(f"  数据形状: obs={obs.shape}, act={act.shape}, 耗时: {elapsed:.2f}s")

    # ---- 检查方块是否被成功搬运 ----
    block_traj = np.array(block_traj)
    block_start = block_traj[0]
    block_end = block_traj[-1]
    displacement = np.linalg.norm(block_end[:2] - block_start[:2])

    print(f"\n[result] 方块轨迹:")
    print(f"  起始: {block_start}")
    print(f"  结束: {block_end}")
    print(f"  水平位移: {displacement:.3f}m (目标: {np.linalg.norm(target_offset[:2]):.3f}m)")

    # 检查各阶段方块 Z 位置
    print(f"\n[result] 各阶段方块 Z 位置:")
    cum_start = 0
    for p in phases:
        cum_end = cum_start + p["hold_steps"]
        seg = block_traj[cum_start:cum_end]
        print(f"  {p['name']:12s} (frames {cum_start}-{cum_end-1}): "
              f"Z mean={seg[:,2].mean():.4f}, min={seg[:,2].min():.4f}, max={seg[:,2].max():.4f}")
        cum_start = cum_end

    if displacement > 0.05:
        print(f"\n  ✅ 方块被成功搬运！")
    elif displacement > 0.02:
        print(f"\n  ⚠ 方块部分搬运（{displacement*100:.0f}cm），抓取不够稳定")
    else:
        print(f"\n  ❌ 方块未被成功抓取，请检查抓取阶段")

    # ---- 保存数据 ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, observations=obs, actions=act)
    print(f"\n[save] ✅ 数据已保存: {output_path}")
    print(f"        obs={obs.shape}, act={act.shape}")

    # ---- Viewer 可视化 ----
    if not args.no_viewer:
        run_viewer_playback(model, data, body_id, joint_traj)

    print("\n👋 完成")


if __name__ == "__main__":
    main()
