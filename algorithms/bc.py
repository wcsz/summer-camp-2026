"""行为克隆 (Behavior Cloning) 的 PyTorch 实现。

这是一个骨架代码，你需要根据课题需求填充具体的网络结构和训练逻辑。

典型用法:
    from algorithms.bc import BehaviorCloning
    model = BehaviorCloning(obs_dim=10, act_dim=6, hidden_dim=256)
    model.train(dataset, epochs=100, batch_size=64)
    model.save("checkpoints/bc_model.pth")
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class MLPPolicy(nn.Module):
    """简单的 MLP 策略网络。

    Args:
        obs_dim: 观测维度
        act_dim: 动作维度
        hidden_dim: 隐藏层宽度 (默认 256)
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class BehaviorCloning:
    """行为克隆训练器。

    Args:
        obs_dim: 观测维度
        act_dim: 动作维度
        hidden_dim: 隐藏层宽度
        lr: 学习率
        device: 训练设备 ('cuda' or 'cpu')
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 lr: float = 1e-3, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.policy = MLPPolicy(obs_dim, act_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        print(f"[BC] 设备: {self.device}, 参数量: {sum(p.numel() for p in self.policy.parameters()):,}")

    def train(self, observations: np.ndarray, actions: np.ndarray,
              epochs: int = 100, batch_size: int = 64, val_split: float = 0.1):
        """训练策略网络。

        Args:
            observations: (N, obs_dim) 观测数组
            actions: (N, act_dim) 动作数组
            epochs: 训练轮数
            batch_size: 批次大小
            val_split: 验证集比例
        """
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        act_tensor = torch.FloatTensor(actions).to(self.device)

        # 划分训练/验证集
        n_val = int(len(observations) * val_split)
        idx = torch.randperm(len(observations))
        train_idx, val_idx = idx[n_val:], idx[:n_val]

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # 训练
            self.policy.train()
            perm = torch.randperm(len(train_idx))
            epoch_loss = 0.0
            n_batches = 0

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
                n_batches += 1

            train_loss = epoch_loss / n_batches

            # 验证
            self.policy.eval()
            with torch.no_grad():
                val_pred = self.policy(obs_tensor[val_idx])
                val_loss = self.loss_fn(val_pred, act_tensor[val_idx]).item()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:>4}/{epochs}  "
                      f"train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}")

        return history

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """用训练好的策略预测动作。"""
        self.policy.eval()
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            return self.policy(obs_tensor).cpu().numpy()

    def save(self, path: str):
        """保存模型权重。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"[BC] 模型已保存: {path}")

    def load(self, path: str):
        """加载模型权重。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[BC] 模型已加载: {path}")
