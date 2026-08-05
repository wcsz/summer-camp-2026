"""训练行为克隆策略。

用法:
    python scripts/train.py                                    # 默认参数
    python scripts/train.py --demo results/demos.npz --epochs 200
"""
import argparse
import numpy as np
import sys
from pathlib import Path

# 将 algorithms 目录加入 Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.bc import BehaviorCloning


def main():
    parser = argparse.ArgumentParser(description='训练行为克隆策略')
    parser.add_argument('--demo', '-d', type=str, default='results/demos.npz',
                        help='demonstration 数据路径')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='训练轮数 (默认: 100)')
    parser.add_argument('--batch_size', '-b', type=int, default=64,
                        help='批次大小 (默认: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率 (默认: 1e-3)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='隐藏层宽度 (默认: 256)')
    parser.add_argument('--output', '-o', type=str, default='checkpoints/bc_model.pth',
                        help='模型保存路径')
    args = parser.parse_args()

    # 1. 加载数据
    demo_path = Path(args.demo)
    if not demo_path.exists():
        print(f"[train] ❌ 数据文件不存在: {demo_path}")
        print(f"[train] 请先运行 python scripts/collect_demo.py 生成数据")
        return

    data = np.load(demo_path)
    observations = data['observations']
    actions = data['actions']
    print(f"[train] 加载数据: {demo_path}")
    print(f"        obs shape: {observations.shape}, act shape: {actions.shape}")

    # 2. 创建 BC 训练器
    obs_dim = observations.shape[1]
    act_dim = actions.shape[1]
    bc = BehaviorCloning(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
    )

    # 3. 训练
    print(f"[train] 开始训练: epochs={args.epochs}, batch_size={args.batch_size}")
    history = bc.train(observations, actions, epochs=args.epochs, batch_size=args.batch_size)

    # 4. 保存模型
    bc.save(args.output)

    # 5. 打印结果摘要
    final_train = history['train_loss'][-1]
    final_val = history['val_loss'][-1]
    best_val = min(history['val_loss'])
    print(f"\n[train] ✅ 训练完成!")
    print(f"       最终 train_loss: {final_train:.6f}")
    print(f"       最终 val_loss:   {final_val:.6f}")
    print(f"       最佳 val_loss:   {best_val:.6f}")


if __name__ == '__main__':
    main()
