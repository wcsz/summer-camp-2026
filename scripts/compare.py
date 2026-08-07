"""BC vs Diffusion Policy 对比评估。

在相同条件下运行 BC 和 DP 的泛化测试，生成对比表格。

用法:
    python scripts/compare.py                           # 默认 5 episodes 泛化测试
    python scripts/compare.py --episodes 10             # 更多测试
    python scripts/compare.py --no-generalize           # 原位置测试（非泛化）
    python scripts/compare.py --bc-model results/bc_model.pth --dp-model results/dp_model.pth
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from algorithms.bc import MLPPolicy
from algorithms.diffusion_policy import DiffusionPolicy
from eval_bc import run_rollout as bc_rollout
from eval_dp import run_rollout as dp_rollout
from collect_pick_place import build_scene_xml


def print_separator():
    print("─" * 72)


def print_header():
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + "     BC vs Diffusion Policy 对比评估".center(62) + "║")
    print("╚" + "═" * 70 + "╝")
    print()


def evaluate(policy_type, model_obj, episodes, generalize, max_steps):
    """运行多 episode 评估，返回统计结果。"""
    xml_str = build_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    block_id = model.body('block').id

    results = []
    inference_times = []

    for ep in range(episodes):
        data.qpos[9] = 0.0
        data.qpos[10] = 0.0
        data.qpos[11] = 0.0

        if generalize:
            noise = np.random.uniform(-0.02, 0.02, size=2)
            data.qpos[9] = noise[0]
            data.qpos[10] = noise[1]

        mujoco.mj_forward(model, data)

        t_start = time.time()
        if policy_type == 'bc':
            disp, max_lift, lifted, placed, moved_right, dist_to_target, _ = \
                bc_rollout(model, data, model_obj, block_id, max_steps)
        else:
            disp, max_lift, lifted, placed, moved_right, dist_to_target, _ = \
                dp_rollout(model, data, model_obj, block_id, max_steps)
        elapsed = time.time() - t_start

        inference_times.append(elapsed)

        is_strict = disp > 0.10 and lifted and placed and moved_right
        is_target = dist_to_target < 0.05

        results.append({
            'episode': ep + 1,
            'displacement': disp,
            'max_lift': max_lift,
            'lifted': lifted,
            'placed': placed,
            'moved_right': moved_right,
            'dist_to_target': dist_to_target,
            'strict': is_strict,
            'target': is_target,
        })

    return results, inference_times


def print_comparison_table(bc_results, dp_results, bc_times, dp_times):
    """打印详细对比表格。"""
    n = len(bc_results)
    print_separator()
    print(f"{'Episode':<8} {'指标':<12} {'BC':<24} {'Diffusion Policy':<24}")
    print_separator()

    for i in range(n):
        bc = bc_results[i]
        dp = dp_results[i]

        print(f"Episode {bc['episode']}")
        print(f"{'':8} {'位移':<12} {bc['displacement']:<24.3f}m {dp['displacement']:<24.3f}m")
        print(f"{'':8} {'最高Z':<12} {bc['max_lift']:<24.3f}m {dp['max_lift']:<24.3f}m")
        print(f"{'':8} {'抬起':<12} {'✓' if bc['lifted'] else '✗':<24} {'✓' if dp['lifted'] else '✗':<24}")
        print(f"{'':8} {'放下':<12} {'✓' if bc['placed'] else '✗':<24} {'✓' if dp['placed'] else '✗':<24}")
        print(f"{'':8} {'右移':<12} {'✓' if bc['moved_right'] else '✗':<24} {'✓' if dp['moved_right'] else '✗':<24}")
        print(f"{'':8} {'距目标':<12} {bc['dist_to_target']:<24.3f}m {dp['dist_to_target']:<24.3f}m")

        bc_status = "✅ 严格成功" if bc['strict'] else ("🔵 学术成功" if bc['target'] else "❌ 失败")
        dp_status = "✅ 严格成功" if dp['strict'] else ("🔵 学术成功" if dp['target'] else "❌ 失败")
        print(f"{'':8} {'结果':<12} {bc_status:<24} {dp_status:<24}")
        print()

    # ---- 汇总 ----
    print_separator()
    print(f"{'汇总':<20} {'BC':<24} {'Diffusion Policy':<24}")
    print_separator()

    bc_strict = sum(1 for r in bc_results if r['strict'])
    dp_strict = sum(1 for r in dp_results if r['strict'])
    bc_target = sum(1 for r in bc_results if r['target'])
    dp_target = sum(1 for r in dp_results if r['target'])

    bc_disp = np.array([r['displacement'] for r in bc_results])
    dp_disp = np.array([r['displacement'] for r in dp_results])
    bc_dist = np.array([r['dist_to_target'] for r in bc_results])
    dp_dist = np.array([r['dist_to_target'] for r in dp_results])

    print(f"{'严格成功率':<20} {bc_strict}/{n} ({100*bc_strict/n:.0f}%){'':<13} {dp_strict}/{n} ({100*dp_strict/n:.0f}%)")
    print(f"{'学术成功率':<20} {bc_target}/{n} ({100*bc_target/n:.0f}%){'':<13} {dp_target}/{n} ({100*dp_target/n:.0f}%)")
    print(f"{'平均位移':<20} {bc_disp.mean():.3f}m ± {bc_disp.std():.3f}{'':<7} {dp_disp.mean():.3f}m ± {dp_disp.std():.3f}")
    print(f"{'平均距目标':<20} {bc_dist.mean():.3f}m ± {bc_dist.std():.3f}{'':<7} {dp_dist.mean():.3f}m ± {dp_dist.std():.3f}")
    print(f"{'平均推理时间/步':<20} {np.mean(bc_times)/1200*1000:.1f}ms{'':<16} {np.mean(dp_times)/1200*1000:.1f}ms")
    print(f"{'总推理时间':<20} {np.mean(bc_times):.1f}s{'':<20} {np.mean(dp_times):.1f}s")
    print_separator()

    # ---- 结论 ----
    print(f"\n📊 分析:")
    if dp_strict > bc_strict:
        print(f"  ✅ DP 严格成功率更高 ({dp_strict}/{n} vs {bc_strict}/{n})")
    elif bc_strict > dp_strict:
        print(f"  ✅ BC 严格成功率更高 ({bc_strict}/{n} vs {dp_strict}/{n})")
    else:
        print(f"  ⚖️  两者严格成功率相同 ({bc_strict}/{n})")

    if dp_dist.mean() < bc_dist.mean():
        print(f"  ✅ DP 最终位置更接近目标 (平均距目标 {dp_dist.mean():.3f}m vs {bc_dist.mean():.3f}m)")
    else:
        print(f"  ✅ BC 最终位置更接近目标 (平均距目标 {bc_dist.mean():.3f}m vs {dp_dist.mean():.3f}m)")

    # 分析失败模式
    bc_catastrophic = sum(1 for r in bc_results if not r['lifted'] and r['displacement'] > 0.2)
    dp_catastrophic = sum(1 for r in dp_results if not r['lifted'] and r['displacement'] > 0.2)
    if bc_catastrophic > dp_catastrophic:
        print(f"  🔍 BC 灾难性失败（方块弹飞）更多: {bc_catastrophic} vs {dp_catastrophic}")
        print(f"     → DP 的动作块预测提供时间一致性，减少失控力")
    elif dp_catastrophic > bc_catastrophic:
        print(f"  🔍 DP 灾难性失败更多: {dp_catastrophic} vs {bc_catastrophic}")

    time_ratio = np.mean(dp_times) / max(np.mean(bc_times), 1e-6)
    print(f"  ⏱️  DP 推理慢 ~{time_ratio:.0f}× (DP={np.mean(dp_times)/1200*1000:.1f}ms/step, BC={np.mean(bc_times)/1200*1000:.1f}ms/step)")
    print()


def main():
    parser = argparse.ArgumentParser(description="BC vs Diffusion Policy 对比评估")
    parser.add_argument('--episodes', '-n', type=int, default=5,
                        help='评估 episode 数量 (默认: 5)')
    parser.add_argument('--max-steps', type=int, default=1200,
                        help='每个 episode 的最大步数')
    parser.add_argument('--no-generalize', action='store_true',
                        help='禁用泛化测试（使用原始方块位置）')
    parser.add_argument('--bc-model', type=str, default='results/bc_model.pth',
                        help='BC 模型路径')
    parser.add_argument('--dp-model', type=str, default='results/dp_model.pth',
                        help='DP 模型路径')
    args = parser.parse_args()

    generalize = not args.no_generalize

    # ---- 加载模型 ----
    bc_path = Path(args.bc_model)
    dp_path = Path(args.dp_model)

    if not bc_path.exists():
        print(f"错误: BC 模型不存在: {bc_path}")
        print(f"请先运行: python scripts/train_bc.py")
        sys.exit(1)
    if not dp_path.exists():
        print(f"错误: DP 模型不存在: {dp_path}")
        print(f"请先运行: python scripts/train_dp.py")
        sys.exit(1)

    print("加载模型...")
    bc = MLPPolicy.load(str(bc_path))
    dp = DiffusionPolicy.load(str(dp_path))

    # ---- 评估 ----
    mode = "泛化测试 (±2cm 随机扰动)" if generalize else "原位置测试"
    print_header()
    print(f"  测试模式: {mode}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print()

    print("🔵 正在评估 BC...")
    bc_results, bc_times = evaluate('bc', bc, args.episodes, generalize, args.max_steps)

    print("🟠 正在评估 Diffusion Policy...")
    dp_results, dp_times = evaluate('dp', dp, args.episodes, generalize, args.max_steps)

    # ---- 打印对比表 ----
    print_comparison_table(bc_results, dp_results, bc_times, dp_times)

    print("👋 完成")


if __name__ == '__main__':
    main()
