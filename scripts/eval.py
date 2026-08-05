"""评估训练好的策略在 MuJoCo 环境中的表现。

用法:
    python scripts/eval.py                                       # 默认参数，仅数值
    python scripts/eval.py --render                              # 打开可视化窗口
    python scripts/eval.py --render --record results/demo.mp4   # 录制视频
"""
import argparse
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from algorithms.bc import BehaviorCloning


def load_model_and_policy(xml_path: str, checkpoint_path: str):
    """加载 MuJoCo 模型和 BC 策略。"""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 加载 BC 策略（需要知道 obs_dim 和 act_dim）
    # 从数据或手动指定
    obs_dim = model.nq + model.nv
    act_dim = model.nu
    bc = BehaviorCloning(obs_dim=obs_dim, act_dim=act_dim)
    bc.load(checkpoint_path)

    return model, data, bc


def run_episode(model, data, bc, steps: int = 500) -> float:
    """运行一个 episode，返回累计奖励（此处用动作平滑度作为代理指标）。"""
    mujoco.mj_resetData(model, data)
    total_reward = 0.0

    for step in range(steps):
        obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
        action = bc.predict(obs.reshape(1, -1))[0]
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        # 简单的平滑度奖励（鼓励小加速度）
        total_reward += -np.sum(np.square(data.qacc))

    return total_reward


def run_with_viewer(model, data, bc, record_path: str = None):
    """带可视化窗口运行（可选录制视频）。"""
    renderer = None
    frames = []

    if record_path:
        renderer = mujoco.Renderer(model, width=640, height=480)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        viewer.sync()

        step = 0
        while viewer.is_running():
            # 用 BC 策略生成动作
            obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            action = bc.predict(obs.reshape(1, -1))[0]
            data.ctrl[:] = action

            mujoco.mj_step(model, data)
            viewer.sync()

            if record_path and step % 2 == 0:  # 每 2 帧保存一帧
                renderer.update_scene(data)
                frames.append(renderer.render())

            step += 1
            if step >= 1000:
                break

    # 保存视频
    if record_path and frames:
        import imageio
        Path(record_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(record_path, frames, fps=30)
        print(f"[eval] 视频已保存: {record_path}")


def main():
    parser = argparse.ArgumentParser(description='评估 BC 策略')
    parser.add_argument('--scene', '-s', type=str, default='envs/my_scene.xml',
                        help='场景 XML 路径')
    parser.add_argument('--checkpoint', '-c', type=str, default='checkpoints/bc_model.pth',
                        help='模型权重路径')
    parser.add_argument('--render', '-r', action='store_true',
                        help='打开可视化窗口')
    parser.add_argument('--record', type=str, default=None,
                        help='录制视频输出路径 (如 results/demo.mp4)')
    parser.add_argument('--episodes', '-n', type=int, default=5,
                        help='评估 episode 数（仅非渲染模式）')
    args = parser.parse_args()

    # 检查文件
    scene_path = Path(args.scene)
    if not scene_path.exists():
        scene_path = Path.home() / 'mujoco' / 'menagerie' / 'franka_emika_panda' / 'panda.xml'
        print(f"[eval] 场景文件不存在，使用 fallback: {scene_path}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[eval] ❌ 模型文件不存在: {ckpt_path}")
        print(f"[eval] 请先运行 python scripts/train.py 训练模型")
        return

    model, data, bc = load_model_and_policy(str(scene_path), str(ckpt_path))

    if args.render:
        print(f"[eval] 🎮 打开可视化窗口...")
        run_with_viewer(model, data, bc, args.record)
    else:
        print(f"[eval] 运行 {args.episodes} 个 episode...")
        rewards = []
        for ep in range(args.episodes):
            r = run_episode(model, data, bc)
            rewards.append(r)
            print(f"  Episode {ep+1}/{args.episodes}: reward={r:.2f}")
        print(f"[eval] 平均 reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


if __name__ == '__main__':
    main()
