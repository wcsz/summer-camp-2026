"""Diffusion Policy 训练入口。

加载第 2 步的 pick_place_demo.npz，训练 Diffusion Policy，保存模型。

与 train_bc.py 的接口风格一致，方便对比实验。

用法:
    python scripts/train_dp.py                           # 默认参数
    python scripts/train_dp.py --epochs 500 --lr 0.0005  # 自定义超参
    python scripts/train_dp.py --horizon 16              # 更长动作块
    python scripts/train_dp.py --diffusion-steps 200     # 更多扩散步数
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from algorithms.diffusion_policy import DiffusionPolicy


def plot_loss_curve(history, output_path):
    """绘制训练/验证 loss 曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], label='Train Loss', alpha=0.7)
    ax.plot(epochs, history['val_loss'], label='Val Loss', alpha=0.7)
    if len(history['val_loss']) > 0:
        best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
        ax.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5,
                   label='Best Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (noise prediction)')
    ax.set_title('Diffusion Policy Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"[plot] Loss 曲线已保存: {output_path}")


def print_loss_summary(history):
    """打印 loss 摘要。"""
    train = history['train_loss']
    val = history['val_loss']
    print(f"\n[loss] 训练摘要 ({len(train)} epochs):")
    print(f"  Train: {train[0]:.6f} → {train[-1]:.6f}")
    print(f"  Val:   {val[0]:.6f} → {val[-1]:.6f}")
    print(f"  Best val: {min(val):.6f} @ epoch {val.index(min(val))+1}")


def main():
    parser = argparse.ArgumentParser(description="训练 Diffusion Policy")
    parser.add_argument('--data', type=str, default='results/pick_place_demo.npz',
                        help='训练数据 .npz 路径')
    parser.add_argument('--epochs', type=int, default=300,
                        help='最大训练轮数 (默认: 300)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小 (默认: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率 (默认: 1e-3)')
    parser.add_argument('--horizon', type=int, default=8,
                        help='动作块长度 H (默认: 8)')
    parser.add_argument('--diffusion-steps', type=int, default=100,
                        help='训练时的扩散步数 T (默认: 100)')
    parser.add_argument('--inference-steps', type=int, default=50,
                        help='推理时的去噪步数 (默认: 50，≤ --diffusion-steps)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stop 等待轮数 (默认: 50)')
    parser.add_argument('--output', '-o', type=str, default='results/dp_model.pth',
                        help='模型输出路径')
    parser.add_argument('--plot', type=str, default='results/dp_loss.png',
                        help='Loss 曲线图输出路径')
    args = parser.parse_args()

    # ---- 加载数据 ----
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"错误: 数据文件不存在: {data_path}")
        print(f"请先运行: python scripts/collect_pick_place.py --no-viewer")
        sys.exit(1)

    data = np.load(data_path)
    observations = data['observations']  # (1200, 24)
    actions = data['actions']            # (1200, 8)
    print(f"[data] 加载: {data_path}")
    print(f"       obs={observations.shape}, act={actions.shape}")

    # ---- 添加时间特征 [step/N] (与 BC 保持一致) ----
    n_frames = len(observations)
    time_feat = np.linspace(0, 1, n_frames).reshape(-1, 1)
    observations = np.concatenate([observations, time_feat], axis=1)
    print(f"[data] 添加时间特征: obs {observations.shape[1]} 维 (含 step/N)")

    # ---- 训练 ----
    if args.inference_steps > args.diffusion_steps:
        args.inference_steps = args.diffusion_steps  # 不能超过训练步数

    dp = DiffusionPolicy(
        obs_dim=observations.shape[1],
        act_dim=actions.shape[1],
        horizon=args.horizon,
        num_diffusion_steps=args.diffusion_steps,
        num_inference_steps=args.inference_steps,
        lr=args.lr,
    )

    history = dp.train(
        observations, actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    # ---- 保存模型 ----
    dp.save(args.output)

    # ---- 绘制 loss 曲线 + 打印摘要 ----
    plot_loss_curve(history, args.plot)
    print_loss_summary(history)

    # ---- 最终指标 ----
    final_train = history['train_loss'][-1]
    best_val = min(history['val_loss'])
    print(f"\n[result] 训练完成:")
    print(f"  Final train loss: {final_train:.6f}")
    print(f"  Best val loss:    {best_val:.6f}")
    print(f"  Epochs trained:   {len(history['train_loss'])}")
    print(f"  Action horizon:   {args.horizon}")
    print(f"  Diffusion steps:  T_train={args.diffusion_steps}, T_infer={args.inference_steps}")

    print(f"\n👋 模型已保存到 {args.output}，运行 eval_dp.py 进行评估")


if __name__ == '__main__':
    main()
