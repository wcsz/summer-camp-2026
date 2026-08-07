"""Diffusion Policy 闭环评估。

加载训练好的 Diffusion Policy，在 MuJoCo 中运行闭环 rollout，评估：
    - 方块水平位移（核心指标）
    - 抬起 / 放下 / 右移 / 距目标距离
    - 泛化能力（改变方块初始位置）
    - 可选录制视频

与 eval_bc.py 使用完全相同的评估框架，确保可比性。

用法:
    python scripts/eval_dp.py                          # 默认评估 + viewer
    python scripts/eval_dp.py --no-viewer --record     # 评估 + 录屏
    python scripts/eval_dp.py --generalize --episodes 5 # 泛化测试
"""

import argparse
import sys
import time
from pathlib import Path

import imageio
import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from algorithms.diffusion_policy import DiffusionPolicy
# 复用 collect_pick_place 的场景构建
from collect_pick_place import build_scene_xml


def run_rollout(model, data, dp, block_id, max_steps=1200, exec_horizon=4):
    """用 Diffusion Policy 运行一次闭环 rollout（receding horizon 控制）。

    每次采样 H 步动作块，执行前 exec_horizon 步，然后重新规划。
    这是 Diffusion Policy 论文的标准做法：H=8 预测, T_a=4 执行。

    Args:
        model, data: MuJoCo 模型和数据
        dp: 加载好的 DiffusionPolicy
        block_id: 方块 body 的 ID
        max_steps: 最大仿真步数
        exec_horizon: 每次采样后执行的步数 (≤ horizon)

    Returns:
        displacement, max_lift, lifted, placed, moved_right, dist_to_target, block_traj
    """
    home_q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
    data.qpos[:9] = home_q
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

    block_start = data.xpos[block_id].copy()
    block_traj = []

    step = 0
    action_chunk = None
    chunk_offset = dp.horizon  # 触发立即采样

    while step < max_steps:
        # 需要采样新动作块？
        if chunk_offset >= exec_horizon:
            raw_obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            time_feat = np.array([step / max_steps])
            obs = np.concatenate([raw_obs, time_feat])
            action_chunk = dp.sample(obs)  # (horizon, act_dim)
            chunk_offset = 0

        # 执行当前块中的动作
        act = action_chunk[chunk_offset]
        data.ctrl[:7] = np.clip(act[:7], -3.0, 3.0)
        data.ctrl[7] = max(0.0, min(255.0, act[7]))
        mujoco.mj_step(model, data)
        block_traj.append(data.xpos[block_id].copy())
        step += 1
        chunk_offset += 1

    block_traj = np.array(block_traj)
    block_end = block_traj[-1]
    block_start_xy = block_start[:2]
    block_end_xy = block_end[:2]

    displacement = float(np.linalg.norm(block_end_xy - block_start_xy))
    max_lift = float(block_traj[:, 2].max())

    lifted = max_lift > block_start[2] + 0.03
    placed = abs(block_end[2] - block_start[2]) < 0.05
    moved_right = block_end_xy[1] > block_start_xy[1] + 0.02

    target_xy = np.array([0.4, 0.15])
    dist_to_target = float(np.linalg.norm(block_end_xy - target_xy))

    return displacement, max_lift, lifted, placed, moved_right, dist_to_target, block_traj


def run_rollout_with_render(model, data, dp, block_id, video_path=None,
                            max_steps=1200, fps=30):
    """运行 rollout 并可选录制视频。"""
    home_q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
    data.qpos[:9] = home_q
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

    block_start = data.xpos[block_id].copy()
    video_frames = []
    renderer = None

    if video_path:
        renderer = mujoco.Renderer(model, width=640, height=480)

    for step in range(max_steps):
        raw_obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
        time_feat = np.array([step / max_steps])
        obs = np.concatenate([raw_obs, time_feat])

        action_chunk = dp.sample(obs)  # (horizon, act_dim)
        act = action_chunk[0]

        data.ctrl[:7] = np.clip(act[:7], -3.0, 3.0)
        data.ctrl[7] = max(0.0, min(255.0, act[7]))

        mujoco.mj_step(model, data)

        if renderer:
            renderer.update_scene(data, camera='record_cam')
            video_frames.append(renderer.render().copy())

    if renderer:
        imageio.mimsave(video_path, video_frames, fps=fps)
        renderer.close()
        print(f"[video] 已保存: {video_path}")

    block_end = data.xpos[block_id].copy()
    displacement = float(np.linalg.norm(block_end[:2] - block_start[:2]))
    return displacement


def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy 闭环评估")
    parser.add_argument('--model', '-m', type=str, default='results/dp_model.pth',
                        help='DP 模型路径')
    parser.add_argument('--episodes', '-n', type=int, default=1,
                        help='评估 episode 数量')
    parser.add_argument('--max-steps', type=int, default=1200,
                        help='每个 episode 的最大步数')
    parser.add_argument('--exec-horizon', type=int, default=4,
                        help='每次采样后执行的步数 (默认: 4, Diffusion Policy 标准做法)')
    parser.add_argument('--generalize', '-g', action='store_true',
                        help='泛化测试: 方块初始位置加 ±2cm 随机扰动')
    parser.add_argument('--no-viewer', action='store_true',
                        help='不显示 viewer')
    parser.add_argument('--record', action='store_true',
                        help='录制 rollout 视频')
    parser.add_argument('--video', type=str, default='results/dp_rollout.mp4',
                        help='视频输出路径')
    args = parser.parse_args()

    # ---- 加载模型 ----
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        print(f"请先运行: python scripts/train_dp.py")
        sys.exit(1)

    dp = DiffusionPolicy.load(str(model_path))

    # ---- 加载场景 ----
    xml_str = build_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    block_id = model.body('block').id

    print(f"\n[eval] Diffusion Policy 评估")
    print(f"  模型: {model_path}")
    print(f"  Episodes: {args.episodes}")
    print(f"  泛化测试: {'是' if args.generalize else '否'}")
    print(f"  推理步数: {dp.num_inference_steps}")

    # ---- 评估 ----
    displacements = []
    success_flags = []
    target_flags = []
    first_noise = None

    for ep in range(args.episodes):
        # 重置方块状态
        data.qpos[9] = 0.0
        data.qpos[10] = 0.0
        data.qpos[11] = 0.0
        if args.generalize:
            noise = np.random.uniform(-0.02, 0.02, size=2)
            if first_noise is None:
                first_noise = noise.copy()
            data.qpos[9] = noise[0]
            data.qpos[10] = noise[1]
            mujoco.mj_forward(model, data)
            block_world_xy = data.xpos[block_id][:2]
            print(f"  Episode {ep+1}: block_world_xy=({block_world_xy[0]:.3f}, {block_world_xy[1]:.3f})")

        if args.record and ep == 0:
            disp = run_rollout_with_render(
                model, data, dp, block_id,
                video_path=args.video, max_steps=args.max_steps,
            )
            lifted = placed = moved_right = False
            max_lift = 0.0
            dist_to_target = 999.0
        else:
            t_start = time.time()
            disp, max_lift, lifted, placed, moved_right, dist_to_target, _ = run_rollout(
                model, data, dp, block_id, max_steps=args.max_steps,
                exec_horizon=args.exec_horizon,
            )
            elapsed = time.time() - t_start
            if ep == 0:
                print(f"  (首次 rollout 耗时: {elapsed:.1f}s, ~{elapsed/args.max_steps*1000:.0f}ms/step)")

        displacements.append(disp)

        is_strict = disp > 0.10 and lifted and placed and moved_right
        is_target = dist_to_target < 0.05
        success_flags.append(is_strict)
        target_flags.append(is_target)

        checks = []
        checks.append(f"位移={disp:.3f}m {'✓' if disp>0.10 else '✗'}")
        checks.append(f"抬起 {'✓' if lifted else '✗'}(max_z={max_lift:.3f}m)")
        checks.append(f"放下 {'✓' if placed else '✗'}")
        checks.append(f"右移 {'✓' if moved_right else '✗'}")
        checks.append(f"距目标={dist_to_target:.3f}m {'✓' if is_target else '✗'}")
        status = "✅" if is_strict else ("🔵" if is_target else "❌")
        print(f"  Episode {ep+1}: {' | '.join(checks)} {status}")

    # ---- 汇总 ----
    disp_arr = np.array(displacements)
    n_strict = sum(success_flags)
    n_target = sum(target_flags)
    print(f"\n[result] 评估汇总:")
    print(f"  平均位移:    {disp_arr.mean():.3f}m ± {disp_arr.std():.3f}")
    print(f"  严格成功(四项全过):        {n_strict}/{len(displacements)}"
          f" ({100*n_strict/max(1,len(displacements)):.0f}%)")
    print(f"  学术标准(距目标<5cm):      {n_target}/{len(displacements)}"
          f" ({100*n_target/max(1,len(displacements)):.0f}%)")
    if n_target > n_strict:
        print(f"  (🔵 = 距目标<5cm 但严格标准未全过)")

    # ---- Viewer（可选） ----
    if not args.no_viewer and not args.record:
        mode_str = f"泛化测试 (episode 1, 位移={displacements[0]:.3f}m)" if args.generalize else "原位置"
        print(f"\n[viewer] 启动交互式 viewer — {mode_str}")
        mujoco.mj_resetData(model, data)
        home_q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
        data.qpos[:9] = home_q
        data.ctrl[:] = 0
        if args.generalize and first_noise is not None:
            data.qpos[9] = first_noise[0]
            data.qpos[10] = first_noise[1]
        mujoco.mj_forward(model, data)

        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                step = 0
                finished = False
                while viewer.is_running():
                    try:
                        if not finished:
                            raw_obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
                            time_feat = np.array([min(step, args.max_steps - 1) / args.max_steps])
                            obs = np.concatenate([raw_obs, time_feat])
                            action_chunk = dp.sample(obs)
                            act = action_chunk[0]
                            data.ctrl[:7] = act[:7]
                            data.ctrl[7] = max(0.0, min(255.0, act[7]))
                            mujoco.mj_step(model, data)
                            step += 1
                            if step >= args.max_steps:
                                finished = True
                                print(f"  [viewer] rollout 完成 (共 {step} 步)，关闭窗口退出")
                        viewer.sync()
                    except Exception:
                        break
        except Exception:
            pass

    print("\n👋 完成")


if __name__ == '__main__':
    main()
