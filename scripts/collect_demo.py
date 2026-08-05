"""收集 expert demonstration 数据用于行为克隆训练。

思路：
    用预设轨迹（如正弦波）或手动控制，驱动 MuJoCo 机械臂完成任务，
    同时记录 (observation, action) 对，保存为 .npz 文件。

用法:
    python scripts/collect_demo.py                          # 默认参数
    python scripts/collect_demo.py --num_episodes 50        # 采集 50 条轨迹
"""
import argparse
import numpy as np
import mujoco
from pathlib import Path


def load_model():
    """加载 MuJoCo 模型。

    直接使用绝对路径加载 menagerie 中的模型。
    避免 <include> 嵌套导致的 meshdir 路径问题。
    如需使用自定义场景，将 xml_path 改为你的 XML 路径。
    """
    # 默认：直接加载 Panda 机械臂
    xml_path = str(Path.home() / 'mujoco' / 'menagerie' / 'franka_emika_panda' / 'panda.xml')

    # 备选：加载自定义场景（注意：如果场景 include 了其他模型，
    # 需要确保 mesh 文件的路径正确，最简单的方式是直接加载原始模型）
    # xml_path = str(Path(__file__).parent.parent / 'envs' / 'my_scene.xml')

    print(f"[collect] 加载模型: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data


def collect_trajectory(model, data, steps: int = 200) -> tuple:
    """运行一条 trajectory 并记录观测和动作。

    Args:
        model: MjModel
        data: MjData
        steps: 每条轨迹的步数

    Returns:
        observations: (steps, obs_dim) numpy array
        actions: (steps, act_dim) numpy array
    """
    observations = []
    actions = []

    for step in range(steps):
        # --- 1. 读取观测 ---
        # 可扩展：加入末端位姿、图像等
        obs = np.concatenate([
            data.qpos.copy(),   # 关节位置
            data.qvel.copy(),   # 关节速度
        ])
        observations.append(obs)

        # --- 2. 生成动作（这里用正弦波做演示，实际应替换为你的控制策略）---
        action = np.sin(step * 0.02 + np.arange(model.nu) * 0.5)
        data.ctrl[:] = action
        actions.append(action.copy())

        # --- 3. 步进仿真 ---
        mujoco.mj_step(model, data)

    return np.array(observations), np.array(actions)


def main():
    parser = argparse.ArgumentParser(description='收集 expert demonstration')
    parser.add_argument('--num_episodes', '-n', type=int, default=10,
                        help='采集轨迹数量 (默认: 10)')
    parser.add_argument('--steps', '-s', type=int, default=200,
                        help='每条轨迹步数 (默认: 200)')
    parser.add_argument('--output', '-o', type=str, default='results/demos.npz',
                        help='输出文件路径')
    args = parser.parse_args()

    model, data = load_model()

    print(f"[collect] 采集 {args.num_episodes} 条轨迹, 每条 {args.steps} 步")
    print(f"[collect] obs_dim={model.nq + model.nv}, act_dim={model.nu}")

    all_obs = []
    all_act = []

    for ep in range(args.num_episodes):
        mujoco.mj_resetData(model, data)
        obs, act = collect_trajectory(model, data, args.steps)
        all_obs.append(obs)
        all_act.append(act)

        if (ep + 1) % 5 == 0:
            print(f"  已采集 {ep+1}/{args.num_episodes} 条轨迹")

    # 合并并保存
    all_obs = np.concatenate(all_obs, axis=0)
    all_act = np.concatenate(all_act, axis=0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, observations=all_obs, actions=all_act)

    print(f"[collect] ✅ 已保存: {output_path}")
    print(f"           数据形状: obs={all_obs.shape}, act={all_act.shape}")


if __name__ == '__main__':
    main()
