"""让 Panda 机械臂末端在三维空间中画出圆形轨迹。

核心技术点：
    1. 笛卡尔空间 → 关节空间的映射（逆运动学 / IK）
    2. 雅可比矩阵伪逆法求解冗余机械臂 IK
    3. MuJoCo 仿真循环与 viewer 交互

用法:
    python scripts/draw_circle.py                  # 默认参数
    python scripts/draw_circle.py --radius 0.15    # 自定义半径
    python scripts/draw_circle.py --center 0.4 0.05 0.45  # 自定义圆心

控制说明（viewer 窗口中）:
    鼠标左键拖动  → 旋转视角
    鼠标滚轮      → 缩放
    Ctrl+鼠标左键 → 平移视角
    Tab           → 切换视野跟踪模式
    Space         → 暂停/继续
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ==============================================================================
# 1. 模型加载
# ==============================================================================

def load_model():
    """加载 Panda (无手) 模型。

    选用 panda_nohand.xml 的原因：
    - 7 个旋转关节，无夹爪手指（夹爪对画圆任务无贡献）
    - 在 link7 末端定义了 attachment_site，正好作为末端执行器参考点
    - 尺寸更小，IK 求解更快
    """
    xml_path = str(
        Path.home() / "mujoco" / "menagerie" / "franka_emika_panda" / "panda_nohand.xml"
    )
    print(f"[draw_circle] 加载模型: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data


# ==============================================================================
# 2. 圆形轨迹生成（笛卡尔空间）
# ==============================================================================

def generate_circle_waypoints(center, radius, normal, num_points=200):
    """在三维空间中生成圆形轨迹的离散采样点。

    核心数学:
        给定圆心 c、半径 r、法向量 n（圆所在平面的法线方向）:
        1. 用 Gram-Schmidt 正交化构造平面内两个正交方向 u, v
        2. 参数方程: P(θ) = c + r * (cos(θ) * u + sin(θ) * v)

    Args:
        center: (3,) 圆心坐标 [x, y, z]
        radius: 半径 (m)
        normal: (3,) 平面法向量（不需要单位化）
        num_points: 采样点数

    Returns:
        waypoints: (num_points, 3) 采样点数组
    """
    c = np.asarray(center, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)  # 单位化法向量

    # Gram-Schmidt: 从一个不与 n 共线的向量出发，构造 u
    # 选 [1, 0, 0]，如果它与 n 太接近就改用 [0, 1, 0]
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary, n)) > 0.99:
        arbitrary = np.array([0.0, 1.0, 0.0])

    u = arbitrary - np.dot(arbitrary, n) * n  # 投影到平面内
    u = u / np.linalg.norm(u)                  # 单位化

    v = np.cross(n, u)  # 叉积：第三个正交方向
    v = v / np.linalg.norm(v)

    theta = np.linspace(0, 2 * np.pi, num_points)
    # P(θ) = c + r * (cosθ * u + sinθ * v)
    waypoints = c + radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)

    return waypoints


# ==============================================================================
# 3. 逆运动学求解器（核心算法）
# ==============================================================================

def solve_ik(
    model,
    data,
    site_id,
    target_pos,
    init_q=None,
    regularization=0.1,
    max_iter=200,
    tol=1e-4,
    verbose=False,
):
    """用阻尼最小二乘法（Damped Least Squares）求解逆运动学。

    === 背景知识 ===

    正向运动学 (FK):  q → x    (关节角度 → 末端位姿)  —— 有唯一解
    逆运动学   (IK):  x → q    (末端位姿 → 关节角度)  —— 可能无解/多解

    Panda 有 7 个自由度，末端位姿约束只有 3 个（位置），因此有 4 个冗余自由度。
    这是"冗余机械臂"——同一个末端位置可以通过无穷多种关节配置达到。

    === 算法原理 ===

    雅可比矩阵 J ∈ ℝ^(3×7) 建立了关节速度与末端速度的线性关系:
        J(q) · q̇ = ẋ

    要"追上"末端位置误差 Δx = target - current，需要:
        Δq = J⁺ · Δx
    其中 J⁺ 是 J 的伪逆。

    但直接用伪逆有两个问题:
    1. 在奇异构型附近，J 接近不满秩，伪逆的数值不稳定（Δq 爆炸）
    2. 没有机制保持关节角度在合理范围

    解决方案——阻尼最小二乘法 (Levenberg-Marquardt):
        Δq = Jᵀ (J Jᵀ + λI)⁻¹ Δx

    其中 λ 是阻尼系数:
    - λ 大 → Δq 小，收敛慢但稳定（"谨慎步进"）
    - λ 小 → Δq 大，收敛快但可能在奇异点附近抖动

    Args:
        model: MuJoCo MjModel
        data: MuJoCo MjData（其 qpos 会被修改）
        site_id: 末端 site 的 ID
        target_pos: (3,) 目标位置 [x, y, z]
        init_q: 初始关节角度（None 则从 data.qpos 出发）
        regularization: 阻尼系数 λ
        max_iter: 最大迭代次数
        tol: 收敛容差 (m)
        verbose: 是否打印每次迭代的误差

    Returns:
        q: (7,) 关节角度解
        success: bool 是否收敛
    """
    if init_q is not None:
        data.qpos[:] = init_q

    mujoco.mj_forward(model, data)

    for i in range(max_iter):
        # --- 当前位置误差 ---
        current_pos = data.site_xpos[site_id]
        error = target_pos - current_pos
        err_norm = float(np.linalg.norm(error))

        if verbose and i % 20 == 0:
            print(f"    IK iter {i:3d}: error = {err_norm:.6f} m")

        if err_norm < tol:
            if verbose:
                print(f"    IK converged at iter {i}, error = {err_norm:.6f} m")
            return data.qpos.copy(), True

        # --- 计算平移雅可比 J_p (3×nv) ---
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        # --- 阻尼最小二乘: Δq = Jᵀ (J Jᵀ + λI)⁻¹ Δx ---
        J = jacp  # 只用平移部分 (3×7)
        JJT = J @ J.T  # (3×3)
        damped = JJT + regularization * np.eye(3)
        delta_q = J.T @ np.linalg.solve(damped, error)

        # --- 更新关节角 ---
        data.qpos[:] += delta_q

        # --- 钳制到关节限位 ---
        for j in range(model.njnt):
            qpos_adr = model.jnt_qposadr[j]
            if model.jnt_limited[j]:
                lo = model.jnt_range[j][0]
                hi = model.jnt_range[j][1]
                data.qpos[qpos_adr] = np.clip(data.qpos[qpos_adr], lo, hi)

        # --- 更新正向运动学 ---
        mujoco.mj_forward(model, data)

    # 达到最大迭代次数
    err_norm = float(np.linalg.norm(target_pos - data.site_xpos[site_id]))
    success = err_norm < tol * 10  # 放宽 10 倍也算"大致成功"
    return data.qpos.copy(), success


# ==============================================================================
# 4. 预计算整条关节轨迹
# ==============================================================================

def precompute_trajectory(model, data, site_id, waypoints, home_q, verbose=True):
    """对每个圆形采样点求解 IK，得到完整的关节空间轨迹。

    关键技巧——热启动 (warm start):
        每个采样点的 IK 求解以上一个点的解作为初始猜测。
        这保证了解的一致性（避免在不同冗余解之间跳变），
        同时显著加速收敛（相邻点的解通常很接近）。

    Args:
        model: MjModel
        data: MjData
        site_id: 末端 site ID
        waypoints: (N, 3) 笛卡尔空间圆形采样点
        home_q: (7,) 初始关节角度
        verbose: 是否打印进度

    Returns:
        joint_trajectory: (N, 7) 关节空间轨迹
        success_rate: 成功收敛的比例
    """
    N = len(waypoints)
    joint_trajectory = np.zeros((N, model.nq))
    successes = 0

    if verbose:
        print(f"\n[draw_circle] 预计算 IK 轨迹 ({N} 个采样点)...")
        t_start = time.time()

    # 起点：先移到圆的第一个点
    prev_q = home_q.copy()

    for i in range(N):
        q, ok = solve_ik(
            model, data, site_id, waypoints[i],
            init_q=prev_q,
            regularization=0.1,
            max_iter=200,
            tol=1e-4,
            verbose=False,
        )
        joint_trajectory[i] = q
        prev_q = q  # 热启动：下一个点从我这里出发
        if ok:
            successes += 1

    if verbose:
        elapsed = time.time() - t_start
        rate = successes / N * 100
        print(f"  完成: {successes}/{N} 收敛 ({rate:.1f}%), 耗时 {elapsed:.1f}s")

    return joint_trajectory, successes / N


# ==============================================================================
# 5. 主仿真循环（轨迹回放 + 交互式 Viewer）
# ==============================================================================

def run_viewer(model, data, joint_trajectory, steps_per_waypoint=10):
    """在 MuJoCo viewer 中回放预计算的关节轨迹。

    控制架构:
        Panda 的 actuator 定义了 gainprm（位置增益 kp > 0），
        因此处于 position control 模式。
        data.ctrl[i] = q_target[i]  →  MuJoCo 内部 PD 控制器计算力矩
                                     →  驱动关节到达目标位置

    Args:
        model: MjModel
        data: MjData
        joint_trajectory: (N, 7) 预计算的关节轨迹
        steps_per_waypoint: 每个采样点保持的仿真步数
    """
    N = len(joint_trajectory)
    mujoco.mj_resetData(model, data)

    print(f"\n[draw_circle] 启动 viewer...")
    print(f"  轨迹点数: {N}, 每步帧数: {steps_per_waypoint}")
    print(f"  预期循环时间: {N * steps_per_waypoint * model.opt.timestep:.1f}s")
    print(f"  控制: 鼠标拖动旋转 | 滚轮缩放 | Ctrl+拖动平移 | Space 暂停 | Esc 退出")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 将机械臂移到轨迹起点
        data.ctrl[:] = joint_trajectory[0]
        data.qpos[:] = joint_trajectory[0]
        mujoco.mj_forward(model, data)

        step = 0
        paused = False  # 跟踪暂停状态（Space 键切换）

        while viewer.is_running():
            # 检查是否刚按下 Space（切换暂停）
            # 注: mujoco.viewer.launch_passive 下无法直接读键盘，
            #     这里用持续同步的方式。Space 由 viewer 内部处理。

            if not paused:
                # 当前目标采样点索引（循环）
                waypoint_idx = (step // steps_per_waypoint) % N
                next_idx = (waypoint_idx + 1) % N

                # 在相邻 IK 解之间线性插值（保证运动平滑）
                alpha = (step % steps_per_waypoint) / steps_per_waypoint
                target_q = (
                    1 - alpha
                ) * joint_trajectory[waypoint_idx] + alpha * joint_trajectory[next_idx]

                # --- 发送位置指令 ---
                data.ctrl[:] = target_q

                # --- 仿真一步 ---
                mujoco.mj_step(model, data)

                step += 1

            # --- 同步 viewer ---
            viewer.sync()

            # 控制回放速度（约 60 FPS 视觉刷新率）
            # 注意: viewer.sync() 内部已有 vsync，这里不需要额外 sleep


# ==============================================================================
# 6. 命令行入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="让 Panda 机械臂末端在三维空间中画圆",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/draw_circle.py                          # 默认参数
  python scripts/draw_circle.py --radius 0.12            # 改变半径
  python scripts/draw_circle.py --center 0.3 0 0.4       # 改变圆心
  python scripts/draw_circle.py --num-points 300          # 更多采样点(更平滑)
  python scripts/draw_circle.py --steps-per 15            # 更慢的运动
        """,
    )
    parser.add_argument(
        "--center", "-c", nargs=3, type=float, default=[0.4, 0.0, 0.5],
        metavar=("X", "Y", "Z"),
        help="圆心坐标 (默认: 0.4 0 0.5)",
    )
    parser.add_argument(
        "--radius", "-r", type=float, default=0.10,
        help="圆半径 (m, 默认: 0.10)",
    )
    parser.add_argument(
        "--normal", "-n", nargs=3, type=float, default=[0.0, 1.0, 0.0],
        metavar=("NX", "NY", "NZ"),
        help="圆平面法向量 (默认: 0 1 0, 即 XZ 平面)",
    )
    parser.add_argument(
        "--num-points", type=int, default=200,
        help="圆形采样点数量 (默认: 200)",
    )
    parser.add_argument(
        "--steps-per", type=int, default=8,
        help="每个采样点的仿真步数 (默认: 8)",
    )
    parser.add_argument(
        "--ik-regularization", type=float, default=0.1,
        help="IK 阻尼系数 λ (默认: 0.1)",
    )

    args = parser.parse_args()

    center = np.array(args.center, dtype=np.float64)
    normal = np.array(args.normal, dtype=np.float64)

    # ---- 加载模型 ----
    model, data = load_model()
    site_id = model.site("attachment_site").id

    # Panda 的 home 配置（来自 panda.xml 的 <keyframe name="home">）
    home_q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

    print(f"[draw_circle] 模型信息:")
    print(f"  DOF: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"  关节范围: ")
    for j in range(model.njnt):
        name = model.joint(j).name
        qpos_adr = model.jnt_qposadr[j]
        if model.jnt_limited[j]:
            lo, hi = model.jnt_range[j]
            print(f"    {name}: [{lo:.2f}, {hi:.2f}]")
        else:
            print(f"    {name}: unlimited")

    # ---- 生成圆形轨迹 ----
    print(f"\n[draw_circle] 圆形参数:")
    print(f"  圆心: {center}")
    print(f"  半径: {args.radius} m")
    print(f"  法向量: {normal}")
    print(f"  采样点: {args.num_points}")

    waypoints = generate_circle_waypoints(center, args.radius, normal, args.num_points)

    # 验证圆心到基座的距离是否在工作空间内
    base_origin = np.zeros(3)
    dist_to_center = np.linalg.norm(center - base_origin)
    panda_reach = 0.855  # Panda 标称工作半径
    max_circle_dist = dist_to_center + args.radius
    print(f"  圆心距基座: {dist_to_center:.3f} m (Panda reach: ~{panda_reach} m)")
    if max_circle_dist > panda_reach * 0.95:
        print(f"  ⚠ 警告: 圆上最远点距基座 {max_circle_dist:.3f}m，"
              f"可能超出工作空间！")
    elif dist_to_center > panda_reach:
        print(f"  ⚠ 警告: 圆心已超出 Panda 标称 reach！")

    # ---- 预计算 IK ----
    joint_traj, success_rate = precompute_trajectory(
        model, data, site_id, waypoints, home_q, verbose=True,
    )

    if success_rate < 0.5:
        print(f"\n⚠ IK 成功率仅 {success_rate*100:.0f}%！")
        print(f"  建议: 减小半径、调整圆心位置、或增大 --ik-regularization")
        if input("  是否继续? (y/N): ").strip().lower() != "y":
            sys.exit(1)

    # ---- 视觉反馈：计算 IK 轨迹对应的实际末端位置误差 ----
    print(f"\n[draw_circle] IK 质量评估:")
    errors = []
    for i in range(len(joint_traj)):
        data.qpos[:] = joint_traj[i]
        mujoco.mj_forward(model, data)
        ee_pos = data.site_xpos[site_id]
        err = np.linalg.norm(ee_pos - waypoints[i])
        errors.append(err)
    errors = np.array(errors)
    print(f"  末端位置误差: mean={errors.mean():.4f}m, max={errors.max():.4f}m, "
          f"median={np.median(errors):.4f}m")

    # ---- 启动 viewer ----
    run_viewer(model, data, joint_traj, steps_per_waypoint=args.steps_per)


if __name__ == "__main__":
    main()
