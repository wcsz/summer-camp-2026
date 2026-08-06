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
        # 构造观测（含时间特征 [step/N]）
        raw_obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
        time_feat = np.array([step / max_steps])
        obs = np.concatenate([raw_obs, time_feat])

        # 策略推理
        act = policy.predict(obs)

        # 执行
        data.ctrl[:7] = act[:7]
        data.ctrl[7] = max(0.0, min(255.0, act[7]))  # 钳制 gripper

        mujoco.mj_step(model, data)
        block_traj.append(data.xpos[block_id].copy())

    block_end = data.xpos[block_id].copy()
    displacement = float(np.linalg.norm(block_end[:2] - block_start[:2]))
    return displacement, np.array(block_traj)


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
    return displacement


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
    for ep in range(args.episodes):
        # 泛化测试: 扰动方块初始位置
        # 方块世界 X = body_pos_X(0.4) + qpos_X，所以要偏移 ±2cm 需设 qpos_X = noise
        if args.generalize:
            noise = np.random.uniform(-0.02, 0.02, size=2)
            data.qpos[9] = noise[0]   # block X 偏移
            data.qpos[10] = noise[1]  # block Y 偏移
            # 计算实际世界位置用于打印
            mujoco.mj_forward(model, data)
            block_world_xy = data.xpos[block_id][:2]
            print(f"  Episode {ep+1}: block_world_xy=({block_world_xy[0]:.3f}, {block_world_xy[1]:.3f})")

        if args.record and ep == 0:
            disp = run_rollout_with_render(
                model, data, policy, block_id,
                video_path=args.video, max_steps=args.max_steps,
            )
        else:
            disp, _ = run_rollout(model, data, policy, block_id, max_steps=args.max_steps)

        displacements.append(disp)
        status = "✅" if disp > 0.10 else ("⚠" if disp > 0.05 else "❌")
        print(f"  Episode {ep+1}: 方块位移={disp:.3f}m {status}")

    # ---- 汇总 ----
    disp_arr = np.array(displacements)
    print(f"\n[result] 评估汇总:")
    print(f"  平均位移:  {disp_arr.mean():.3f}m ± {disp_arr.std():.3f}")
    print(f"  最大位移:  {disp_arr.max():.3f}m")
    print(f"  最小位移:  {disp_arr.min():.3f}m")
    print(f"  成功率:    {(disp_arr > 0.10).sum()}/{len(disp_arr)} "
          f"({100*(disp_arr > 0.10).mean():.0f}%)")

    # ---- Viewer（可选） ----
    if not args.no_viewer and not args.record:
        mode_str = "泛化测试" if args.generalize else "原位置"
        print(f"\n[viewer] 启动交互式 viewer ({mode_str})...")
        mujoco.mj_resetData(model, data)
        home_q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
        data.qpos[:9] = home_q
        data.ctrl[:] = 0
        # 泛化测试: 在 viewer 中也扰动方块位置
        if args.generalize:
            noise = np.random.uniform(-0.02, 0.02, size=2)
            data.qpos[9] = noise[0]
            data.qpos[10] = noise[1]
        mujoco.mj_forward(model, data)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            step = 0
            while viewer.is_running() and step < args.max_steps:
                raw_obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
                time_feat = np.array([step / args.max_steps])
                obs = np.concatenate([raw_obs, time_feat])
                act = policy.predict(obs)
                data.ctrl[:7] = act[:7]
                data.ctrl[7] = max(0.0, min(255.0, act[7]))
                mujoco.mj_step(model, data)
                viewer.sync()
                step += 1
                if step % 300 == 0:
                    print(f"  viewer step {step}/{args.max_steps}")

    print("\n👋 完成")


if __name__ == '__main__':
    main()
