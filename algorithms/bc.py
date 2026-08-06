"""行为克隆 (Behavior Cloning) 的 PyTorch 实现。

包含: MLP 策略网络 + 数据标准化 + BC 训练器 + 模型存取

典型用法:
    from algorithms.bc import BCTrainer, MLPPolicy

    # 训练
    trainer = BCTrainer(obs_dim=24, act_dim=8)
    history = trainer.train(observations, actions, epochs=300)
    trainer.save("results/bc_model.pth")

    # 推理
    model = MLPPolicy.load("results/bc_model.pth")
    action = model.predict(observation)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


# ==============================================================================
# 1. MLP 策略网络
# ==============================================================================

class MLPPolicy(nn.Module):
    """三层 MLP 策略网络: 24 → 256 → 256 → 256 → 8。

    Args:
        obs_dim: 观测维度
        act_dim: 动作维度
        hidden_dim: 隐藏层宽度
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

        # 标准化参数（推理时由 load 填充）
        self.register_buffer('obs_mean', torch.zeros(obs_dim))
        self.register_buffer('obs_std', torch.ones(obs_dim))
        self.register_buffer('act_mean', torch.zeros(act_dim))
        self.register_buffer('act_std', torch.ones(act_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """前向传播（假设输入已标准化）。"""
        return self.net(obs)

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """从原始观测预测原始动作（包含标准化/反标准化）。

        Args:
            obs: (obs_dim,) 或 (batch, obs_dim) 原始观测

        Returns:
            act: 原始动作（与 obs 同 batch 维度）
        """
        self.eval()
        single = obs.ndim == 1
        if single:
            obs = obs.reshape(1, -1)

        obs_tensor = torch.FloatTensor(obs)
        obs_norm = (obs_tensor - self.obs_mean) / (self.obs_std + 1e-8)
        act_norm = self.forward(obs_norm)
        act = act_norm * self.act_std + self.act_mean

        if single:
            return act.numpy().flatten()
        return act.numpy()

    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """从文件加载模型（包含标准化参数）。"""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            obs_dim=ckpt['obs_dim'],
            act_dim=ckpt['act_dim'],
            hidden_dim=ckpt.get('hidden_dim', 256),
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.obs_mean = ckpt['obs_mean'].to(device)
        model.obs_std = ckpt['obs_std'].to(device)
        model.act_mean = ckpt['act_mean'].to(device)
        model.act_std = ckpt['act_std'].to(device)
        model.eval()
        print(f"[BC] 模型已加载: {path} (obs_dim={model.obs_dim}, act_dim={model.act_dim})")
        return model


# ==============================================================================
# 2. BC 训练器
# ==============================================================================

class BCTrainer:
    """行为克隆训练器，包含标准化、训练循环、early stopping。

    Args:
        obs_dim: 观测维度
        act_dim: 动作维度
        hidden_dim: 隐藏层宽度 (默认 256)
        lr: 学习率
        device: 训练设备
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 lr: float = 1e-3, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        self.policy = MLPPolicy(obs_dim, act_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        n_params = sum(p.numel() for p in self.policy.parameters())
        print(f"[BC] 设备: {self.device}")
        print(f"[BC] 网络: {obs_dim}→{hidden_dim}→{hidden_dim}→{hidden_dim}→{act_dim}")
        print(f"[BC] 参数量: {n_params:,}")

        # 标准化参数（训练时计算）
        self.obs_mean = None
        self.obs_std = None
        self.act_mean = None
        self.act_std = None

    def _normalize(self, obs, act):
        """计算并应用标准化。"""
        self.obs_mean = obs.mean(axis=0)
        self.obs_std = obs.std(axis=0) + 1e-8
        self.act_mean = act.mean(axis=0)
        self.act_std = act.std(axis=0) + 1e-8

        obs_norm = (obs - self.obs_mean) / self.obs_std
        act_norm = (act - self.act_mean) / self.act_std

        return obs_norm, act_norm

    def train(self, observations: np.ndarray, actions: np.ndarray,
              epochs: int = 300, batch_size: int = 64, val_split: float = 0.2,
              patience: int = 50):
        """训练 BC 策略。

        Args:
            observations: (N, obs_dim) — 如已含时间特征，obs_dim=25；否则 obs_dim=24
            actions: (N, act_dim)
            epochs: 最大训练轮数
            batch_size: 批次大小
            val_split: 验证集比例
            patience: early stop 等待轮数

        Returns:
            history: dict with 'train_loss' and 'val_loss' lists
        """
        # 标准化数据
        obs_norm, act_norm = self._normalize(observations, actions)
        obs_tensor = torch.FloatTensor(obs_norm).to(self.device)
        act_tensor = torch.FloatTensor(act_norm).to(self.device)

        # 划分训练/验证集
        n = len(observations)
        n_val = int(n * val_split)
        indices = torch.randperm(n)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        print(f"[BC] 数据: {n} 帧, 训练 {n-n_val}, 验证 {n_val}")
        print(f"[BC] Early stop patience: {patience} epochs")

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_epoch = 0
        no_improve = 0

        for epoch in range(epochs):
            # ---- 训练 ----
            self.policy.train()
            perm = torch.randperm(len(train_idx))
            epoch_loss = 0.0
            batches = 0

            for i in range(0, len(train_idx), batch_size):
                batch_idx = perm[i:i + batch_size]
                obs_batch = obs_tensor[train_idx[batch_idx]]
                act_batch = act_tensor[train_idx[batch_idx]]

                pred = self.policy(obs_batch)
                loss = self.loss_fn(pred, act_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            train_loss = epoch_loss / max(batches, 1)

            # ---- 验证 ----
            self.policy.eval()
            with torch.no_grad():
                val_pred = self.policy(obs_tensor[val_idx])
                val_loss = self.loss_fn(val_pred, act_tensor[val_idx]).item()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # ---- Early stop ----
            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                no_improve = 0
                # 保存最佳权重
                self._best_state = {
                    k: v.cpu().clone() for k, v in self.policy.state_dict().items()
                }
            else:
                no_improve += 1

            if (epoch + 1) % 50 == 0 or epoch == 0:
                marker = " *" if no_improve == 0 else ""
                print(f"  Epoch {epoch+1:>4}/{epochs} | "
                      f"train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}{marker}")

            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1} (best: epoch {best_epoch}, "
                      f"val_loss={best_val_loss:.6f})")
                break

        # 恢复最佳权重
        if hasattr(self, '_best_state'):
            self.policy.load_state_dict({k: v.to(self.device) for k, v in self._best_state.items()})

        # 把标准化参数写入模型
        self.policy.obs_mean = torch.FloatTensor(self.obs_mean).to(self.device)
        self.policy.obs_std = torch.FloatTensor(self.obs_std).to(self.device)
        self.policy.act_mean = torch.FloatTensor(self.act_mean).to(self.device)
        self.policy.act_std = torch.FloatTensor(self.act_std).to(self.device)

        return history

    def save(self, path: str):
        """保存模型权重和标准化参数。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'hidden_dim': self.hidden_dim,
            'obs_mean': self.policy.obs_mean.cpu(),
            'obs_std': self.policy.obs_std.cpu(),
            'act_mean': self.policy.act_mean.cpu(),
            'act_std': self.policy.act_std.cpu(),
        }, path)
        print(f"[BC] 模型已保存: {path}")
