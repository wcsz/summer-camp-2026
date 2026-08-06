"""BC 策略闭环评估。

加载训练好的 BC 模型，在 MuJoCo 中运行闭环 rollout，评估：
    - 方块水平位移（核心指标）
    - 泛化能力（改变方块初始位置）
    - 可选录制视频

用法:
    python scripts/eval_bc.py                          # 默认评估 + viewer
    python scripts/eval_bc.py --no-viewer --record     # 评估 + 录屏
    python scripts/eval_bc.py --generalize             # 泛化测试(方块位置随机扰动)
    python scripts/eval_bc.py --episodes 10            # 多次评估取平均
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
from algorithms.bc import MLPPolicy
# 复用 collect_pick_place 的场景构建
from collect_pick_place import build_scene_xml


def run_rollout(model, data, policy, block_id, max_steps=1200):
    """用 BC 策略运行一次闭环 rollout。

    Args:
        model, data: MuJoCo 模型和数据
        policy: 加载好的 MLPPolicy
        block_id: 方块 body 的 ID
        max_steps: 最大仿真步数

    Returns:
        displacement: 方块的水平位移 (m)
        max_lift: 方块达到的最高 Z (m)
        lifted: 方块是否被抬起过（max_lift > init_z + 3cm）
        block_traj: (steps, 3) 方块位置轨迹
    """
    # 初始化
    home_q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
    data.qpos[:9] = home_q
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

    block_start = data.xpos[block_id].copy()
    block_traj = []

    for step in range(max_steps):
        raw_obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
        time_feat = np.array([step / max_steps])
        obs = np.concatenate([raw_obs, time_feat])
        act = policy.predict(obs)
        data.ctrl[:7] = act[:7]
        data.ctrl[7] = max(0.0, min(255.0, act[7]))
        mujoco.mj_step(model, data)
        block_traj.append(data.xpos[block_id].copy())

    block_traj = np.array(block_traj)
    block_end = block_traj[-1]
    block_start_xy = block_start[:2]
    block_end_xy = block_end[:2]

    displacement = float(np.linalg.norm(block_end_xy - block_start_xy))
    max_lift = float(block_traj[:, 2].max())

    # 三标准评判真正的 pick-and-place:
    # 1. 被抬起: Z 最高点 > 初始 Z + 3cm
    lifted = max_lift > block_start[2] + 0.03
    # 2. 被放下: 最终 Z 接近初始高度（±5cm），说明方块被放置而非飞走
    placed = abs(block_end[2] - block_start[2]) < 0.05
    # 3. 方向正确: 最终 Y > 初始 Y + 2cm（目标方向是 Y+=15cm）
    moved_right = block_end_xy[1] > block_start_xy[1] + 0.02

    return displacement, max_lift, lifted, placed, moved_right, block_traj


def run_rollout_with_render(model, data, policy, block_id, video_path=None,
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
        act = policy.predict(obs)

        data.ctrl[:7] = act[:7]
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
    return displacement  # run_rollout_with_render 不返回 lifted 信息


def main():
    parser = argparse.ArgumentParser(description="BC 策略闭环评估")
    parser.add_argument('--model', '-m', type=str, default='results/bc_model.pth',
                        help='BC 模型路径')
    parser.add_argument('--episodes', '-n', type=int, default=1,
                        help='评估 episode 数量')
    parser.add_argument('--max-steps', type=int, default=1200,
                        help='每个 episode 的最大步数')
    parser.add_argument('--generalize', '-g', action='store_true',
                        help='泛化测试: 方块初始位置加 ±2cm 随机扰动')
    parser.add_argument('--no-viewer', action='store_true',
                        help='不显示 viewer')
    parser.add_argument('--record', action='store_true',
                        help='录制 rollout 视频')
    parser.add_argument('--video', type=str, default='results/bc_rollout.mp4',
                        help='视频输出路径')
    args = parser.parse_args()

    # ---- 加载模型 ----
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        print(f"请先运行: python scripts/train_bc.py")
        sys.exit(1)

    policy = MLPPolicy.load(str(model_path))

    # ---- 加载场景 ----
    xml_str = build_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    block_id = model.body('block').id

    print(f"\n[eval] BC 策略评估")
    print(f"  模型: {model_path}")
    print(f"  Episodes: {args.episodes}")
    print(f"  泛化测试: {'是' if args.generalize else '否'}")

    # ---- 评估 ----
    displacements = []
    success_flags = []
    detail_lines = []
    first_noise = None
    for ep in range(args.episodes):
        # 每个 episode 前重置方块状态（避免上一轮的残留）
        data.qpos[9] = 0.0   # block X offset
        data.qpos[10] = 0.0  # block Y offset
        data.qpos[11] = 0.0  # block Z offset
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
                model, data, policy, block_id,
                video_path=args.video, max_steps=args.max_steps,
            )
            lifted = placed = moved_right = False
            max_lift = 0.0
        else:
            disp, max_lift, lifted, placed, moved_right, _ = run_rollout(
                model, data, policy, block_id, max_steps=args.max_steps,
            )

        displacements.append(disp)

        # 四标准评判真正的 pick-and-place:
        # ① 位移 > 10cm  ② 被抬起  ③ 被放下（最终高度接近初始） ④ 方向正确（右移）
        is_success = disp > 0.10 and lifted and placed and moved_right
        success_flags.append(is_success)

        checks = []
        checks.append(f"位移={disp:.3f}m {'✓' if disp>0.10 else '✗'}")
        checks.append(f"抬起 {'✓' if lifted else '✗'}(max_z={max_lift:.3f}m)")
        checks.append(f"放下 {'✓' if placed else '✗'}")
        checks.append(f"右移 {'✓' if moved_right else '✗'}")
        status = "✅" if is_success else "❌"
        print(f"  Episode {ep+1}: {' | '.join(checks)} {status}")

    # ---- 汇总 ----
    disp_arr = np.array(displacements)
    n_success = sum(success_flags)
    print(f"\n[result] 评估汇总:")
    print(f"  平均位移:  {disp_arr.mean():.3f}m ± {disp_arr.std():.3f}")
    print(f"  真成功率(位移+抬起+放下+右移): {n_success}/{len(displacements)}"
          f" ({100*n_success/max(1,len(displacements)):.0f}%)")

    # ---- Viewer（可选） ----
    # viewer 重放第一个 episode 的场景，方便和终端输出的位移数据对照
    if not args.no_viewer and not args.record:
        mode_str = f"泛化测试 (episode 1, 位移={displacements[0]:.3f}m)" if args.generalize else "原位置"
        print(f"\n[viewer] 启动交互式 viewer — {mode_str}")
        mujoco.mj_resetData(model, data)
        home_q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
        data.qpos[:9] = home_q
        data.ctrl[:] = 0
        # 重放第一个 episode 的初始方块位置
        if args.generalize and first_noise is not None:
            data.qpos[9] = first_noise[0]
            data.qpos[10] = first_noise[1]
        mujoco.mj_forward(model, data)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            step = 0
            finished = False
            while viewer.is_running():
                if not finished:
                    raw_obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
                    time_feat = np.array([min(step, args.max_steps - 1) / args.max_steps])
                    obs = np.concatenate([raw_obs, time_feat])
                    act = policy.predict(obs)
                    data.ctrl[:7] = act[:7]
                    data.ctrl[7] = max(0.0, min(255.0, act[7]))
                    mujoco.mj_step(model, data)
                    step += 1
                    if step >= args.max_steps:
                        finished = True
                        print(f"  [viewer] rollout 完成 (共 {step} 步)，viewer 保持打开，按 Esc 退出")
                viewer.sync()

    print("\n👋 完成")


if __name__ == '__main__':
    main()
