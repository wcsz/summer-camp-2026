"""BC 策略训练入口。

加载第 2 步的 pick_place_demo.npz，训练 MLP 行为克隆策略，保存模型。

用法:
    python scripts/train_bc.py                           # 默认参数
    python scripts/train_bc.py --epochs 500 --lr 0.0005  # 自定义超参
    python scripts/train_bc.py --data results/my_demo.npz  # 使用其他数据
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# 把 algorithms 目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from algorithms.bc import BCTrainer


def print_loss_summary(history):
    """打印 loss 曲线摘要（无需 matplotlib）。"""
    train = history['train_loss']
    val = history['val_loss']
    n = len(train)
    print(f"\n[loss] 训练曲线摘要 ({n} epochs):")
    print(f"  Train: {train[0]:.6f} → {train[-1]:.6f}")
    print(f"  Val:   {val[0]:.6f} → {val[-1]:.6f}")
    print(f"  Best val: {min(val):.6f} @ epoch {val.index(min(val))+1}")
    # 简单的 ASCII 曲线
    if n > 1:
        width = 40
        v_min, v_max = min(min(train), min(val)), max(train[0], val[0])
        if v_max > v_min:
            print(f"  Train: {'█' * int((train[0]-train[-1])/(train[0]-v_min+1e-8)*width) if train[-1] < train[0] else '—'}")
            print(f"  Val:   {'█' * int((val[0]-val[-1])/(val[0]-v_min+1e-8)*width) if val[-1] < val[0] else '—'}")


def main():
    parser = argparse.ArgumentParser(description="训练 BC 行为克隆策略")
    parser.add_argument('--data', type=str, default='results/pick_place_demo.npz',
                        help='训练数据 .npz 路径')
    parser.add_argument('--epochs', type=int, default=300,
                        help='最大训练轮数 (默认: 300)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小 (默认: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率 (默认: 1e-3)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='隐藏层宽度 (默认: 256)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stop 等待轮数 (默认: 50)')
    parser.add_argument('--output', '-o', type=str, default='results/bc_model.pth',
                        help='模型输出路径')
    args = parser.parse_args()

    # ---- 加载数据 ----
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"错误: 数据文件不存在: {data_path}")
        print(f"请先运行: python scripts/collect_pick_place.py --no-viewer")
        sys.exit(1)

    data = np.load(data_path)
    observations = data['observations']
    actions = data['actions']
    print(f"[data] 加载: {data_path}")
    print(f"       obs={observations.shape}, act={actions.shape}")

    # ---- 添加时间特征 [step/N] ----
    # 帮助模型感知当前处于轨迹的哪个阶段，缓解 BC 分布偏移问题
    n_frames = len(observations)
    time_feat = np.linspace(0, 1, n_frames).reshape(-1, 1)
    observations = np.concatenate([observations, time_feat], axis=1)
    print(f"[data] 添加时间特征: obs {observations.shape[1]} 维 (含 step/N)")

    # ---- 训练 ----
    trainer = BCTrainer(
        obs_dim=observations.shape[1],
        act_dim=actions.shape[1],
        hidden_dim=args.hidden_dim,
        lr=args.lr,
    )

    history = trainer.train(
        observations, actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    # ---- 保存模型 ----
    trainer.save(args.output)

    # ---- 打印 loss 摘要 ----
    print_loss_summary(history)

    # ---- 最终指标 ----
    final_train = history['train_loss'][-1]
    final_val = history['val_loss'][-1]
    best_val = min(history['val_loss'])
    print(f"\n[result] 训练完成:")
    print(f"  Final train loss: {final_train:.6f}")
    print(f"  Best val loss:    {best_val:.6f}")
    print(f"  Epochs trained:   {len(history['train_loss'])}")

    print(f"\n👋 模型已保存到 {args.output}，运行 eval_bc.py 进行评估")


if __name__ == '__main__':
    main()
