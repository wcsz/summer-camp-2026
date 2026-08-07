"""Diffusion Policy 的 PyTorch 实现。

基于 Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
(RSS 2023) 的核心思想，针对低维观测空间做了 MLP 简化。

核心组件:
    - SinusoidalTimeEmbedding: 扩散时间步 k → 正弦位置编码
    - DDPMScheduler: DDPM 噪声调度器（前向加噪 + 反向去噪）
    - ActionChunkDenoiser: MLP 去噪器（噪声动作块 + 观测 + 时间嵌入 → 预测噪声）
    - DiffusionPolicy: 完整封装（训练/推理/存取）

与 BC 的关键区别:
    - BC 输出确定性的单步动作 a = f(o)
    - DP 输出动作块 [a_t, ..., a_{t+H-1}]，通过去噪采样从 p(a|o) 中生成
    - 动作块提供时间一致性 → 更平滑的轨迹
    - 从分布采样而非取条件期望 → 对分布偏移更鲁棒

典型用法:
    from algorithms.diffusion_policy import DiffusionPolicy

    # 训练
    dp = DiffusionPolicy(obs_dim=25, act_dim=8, horizon=8)
    history = dp.train(observations, actions, epochs=300)
    dp.save("results/dp_model.pth")

    # 推理
    dp = DiffusionPolicy.load("results/dp_model.pth")
    action_chunk = dp.sample(observation)  # (8, 8), 取第一个执行
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


# ==============================================================================
# 1. 正弦时间位置编码
# ==============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """将扩散时间步 k ∈ [0, T-1] 映射为固定频率的正弦位置编码。

    与 Transformer 位置编码原理相同: 不同频率的正弦函数组合，让模型
    通过内积感知时间步之间的距离。

    Args:
        dim: 输出嵌入维度（必须为偶数）
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

        # 频率: 1 / 10000^(2i/dim)，i = 0, 1, ..., dim/2-1
        exponent = torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim)
        self.register_buffer('freq', torch.exp(exponent))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """将时间步张量映射为正弦嵌入。

        Args:
            t: (batch,) int tensor, 扩散时间步

        Returns:
            emb: (batch, dim) float tensor, 正弦位置编码
        """
        # t: (batch,) → (batch, 1)
        t = t.float().unsqueeze(-1)
        # arg: (batch, dim/2)
        arg = t * self.freq.unsqueeze(0)
        emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
        return emb


# ==============================================================================
# 2. DDPM 噪声调度器
# ==============================================================================

class DDPMScheduler:
    """DDPM 噪声调度器——管理扩散过程的前向加噪和反向去噪。

    前向过程 (加噪):
        q(a_k | a_0) = N(√ᾱ_k · a_0, (1-ᾱ_k) · I)
        其中 a_k = √ᾱ_k · a_0 + √(1-ᾱ_k) · ε,  ε ~ N(0, I)

    反向过程 (去噪, 由神经网络 ε_θ 引导):
        p_θ(a_{k-1} | a_k) = N(μ_θ(a_k, k), σ_k² · I)
        其中 μ_θ = (1/√α_k) · (a_k - (β_k/√(1-ᾱ_k)) · ε_θ(a_k, k))

    Args:
        num_steps: 扩散步数 T
        beta_start, beta_end: β schedule 的起止值
        schedule: 'linear' 或 'cosine'
    """

    def __init__(self, num_steps: int = 100, beta_start: float = 1e-4,
                 beta_end: float = 0.02, schedule: str = 'cosine'):
        self.num_steps = num_steps

        if schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_steps)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_steps)

        # 预计算全部系数
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_k, k=0..T-1

        # alphas_cumprod_prev[k] = ᾱ_{k-1}, 其中 ᾱ_{-1} = 1.0
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # 后验方差 σ_k² = β_k · (1-ᾱ_{k-1})/(1-ᾱ_k)
        posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register('betas', self.betas)
        self.register('alphas', alphas)
        self.register('alphas_cumprod', alphas_cumprod)
        self.register('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))          # √ᾱ_k
        self.register('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))  # √(1-ᾱ_k)
        self.register('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))              # 1/√α_k
        self.register('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register('posterior_variance', posterior_variance)

    @staticmethod
    def _cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine β schedule (Nichol & Dhariwal, 2021)。

        相比 linear schedule, cosine 在低噪声区域变化更平缓, 生成质量更好。
        """
        steps = num_steps + 1
        t = torch.linspace(0, num_steps, steps)
        ft = torch.cos((t / num_steps + s) / (1 + s) * torch.pi / 2) ** 2
        betas = torch.clamp(1 - ft[1:] / ft[:-1], min=1e-5, max=0.999)
        return betas

    def register(self, name: str, tensor: torch.Tensor):
        """注册为 buffer（不参与梯度，但随模型移动到 GPU）。"""
        self.register_buffer(name, tensor)

    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)

    def add_noise(self, a_0: torch.Tensor, noise: torch.Tensor,
                  k: torch.Tensor) -> torch.Tensor:
        """前向加噪: a_k = √ᾱ_k · a_0 + √(1-ᾱ_k) · ε。

        Args:
            a_0: (batch, action_dim) 原始干净动作块
            noise: (batch, action_dim) 高斯噪声
            k: (batch,) 扩散时间步

        Returns:
            a_k: (batch, action_dim) 加噪后的动作块
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[k]  # (batch,)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[k]  # (batch,)

        # 扩展维度以支持广播: (batch,) → (batch, 1)
        sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha * a_0 + sqrt_one_minus * noise

    def step(self, model_output: torch.Tensor, k: torch.Tensor,
             a_k: torch.Tensor) -> torch.Tensor:
        """单步反向去噪（DDPM sampling）。

        Args:
            model_output: (batch, action_dim) 神经网络预测的噪声 ε_θ(a_k, k)
            k: (batch,) 当前扩散时间步
            a_k: (batch, action_dim) 当前噪声动作块

        Returns:
            a_{k-1}: (batch, action_dim) 去噪一步后的动作块
        """
        # 用预测噪声计算去噪后的 a_0
        # a_0_hat = (1/√ᾱ_k) · a_k - (√(1/ᾱ_k - 1)) · ε̂
        # 但更稳定的写法是 DDPM 原始公式

        beta_k = self.betas[k].unsqueeze(-1)                     # (batch, 1)
        alpha_k = self.alphas[k].unsqueeze(-1)                   # (batch, 1)
        alpha_cumprod_k = self.alphas_cumprod[k].unsqueeze(-1)  # (batch, 1)

        # 预测 x_0
        sqrt_recip_alpha = 1.0 / torch.sqrt(alpha_k)
        pred_a_0 = sqrt_recip_alpha * a_k - (beta_k / torch.sqrt(1.0 - alpha_cumprod_k)) * model_output

        # 后验均值 μ_θ(a_k, k)
        # μ = (√ᾱ_{k-1}·β_k / (1-ᾱ_k)) · â_0 + (√α_k·(1-ᾱ_{k-1}) / (1-ᾱ_k)) · a_k
        alphas_cumprod_prev = self.alphas_cumprod_prev[k].unsqueeze(-1)  # ᾱ_{k-1}, 其中 ᾱ_{-1}=1.0
        coeff_pred = (torch.sqrt(alphas_cumprod_prev + 1e-8) * beta_k
                      / (1.0 - alpha_cumprod_k))
        coeff_curr = (torch.sqrt(alpha_k + 1e-8) * (1.0 - alphas_cumprod_prev)
                      / (1.0 - alpha_cumprod_k))
        mu = coeff_pred * pred_a_0 + coeff_curr * a_k

        # 加噪声 (k > 0 时)
        noise = torch.randn_like(a_k)
        variance = self.posterior_variance[k].unsqueeze(-1)
        # k=0 时不加噪声
        mask = (k > 0).float().unsqueeze(-1)
        return mu + mask * torch.sqrt(variance + 1e-8) * noise

    def step_ddim(self, model_output: torch.Tensor, k: torch.Tensor,
                  a_k: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        """DDIM 采样步（确定性去噪, η=0 时完全确定）。

        x_0_hat = (a_k - √(1-ᾱ_k) · ε̂) / √ᾱ_k
        a_{k-1} = √ᾱ_{k-1} · x_0_hat + √(1-ᾱ_{k-1} - σ_k²) · ε̂ + σ_k · z

        当 η=0: 确定性 DDIM, σ_k=0
        当 η=1: 等价于 DDPM

        Args:
            model_output: (batch, action_dim) 预测噪声 ε̂
            k: (batch,) 当前扩散时间步
            a_k: (batch, action_dim) 当前噪声动作
            eta: 随机性参数 (0=确定性DDIM, 1=等价DDPM)

        Returns:
            a_{k-1}: (batch, action_dim)
        """
        # x_0 预测
        alpha_cumprod_k = self.alphas_cumprod[k].unsqueeze(-1)   # (batch, 1)
        pred_x0 = (a_k - torch.sqrt(1.0 - alpha_cumprod_k) * model_output) / \
                  torch.sqrt(alpha_cumprod_k + 1e-8)

        # 指向 x_{k-1} 的方向
        alphas_cumprod_prev = self.alphas_cumprod_prev[k].unsqueeze(-1)  # ᾱ_{k-1}

        # DDIM sigma
        sigma_k = eta * torch.sqrt(
            (1.0 - alphas_cumprod_prev) / (1.0 - alpha_cumprod_k + 1e-8) *
            (1.0 - alpha_cumprod_k / (alphas_cumprod_prev + 1e-8))
        )

        # 均值（确定性部分）
        pred_dir = torch.sqrt(1.0 - alphas_cumprod_prev - sigma_k ** 2 + 1e-8) * model_output
        mean = torch.sqrt(alphas_cumprod_prev + 1e-8) * pred_x0 + pred_dir

        # 随机噪声（η=0 时此项为 0）
        noise = torch.randn_like(a_k)
        mask = (k > 0).float().unsqueeze(-1)  # k=0 不加噪
        return mean + mask * sigma_k * noise

    def to(self, device: str):
        """将所有 buffer 移动到指定设备。"""
        for name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                      'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                      'sqrt_recip_alphas', 'sqrt_recip_alphas_cumprod',
                      'posterior_variance']:
            if hasattr(self, name):
                setattr(self, name, getattr(self, name).to(device))
        return self


# ==============================================================================
# 3. 动作块去噪器 (MLP)
# ==============================================================================

class ActionChunkDenoiser(nn.Module):
    """MLP 去噪器: ε_θ(a_k, o, k) → 预测噪声。

    输入为噪声动作块 + 观测 + 时间嵌入的拼接，输出与噪声动作块同维度。

    Args:
        action_dim: 动作块展平维度 (horizon × act_dim)
        obs_dim: 观测维度
        time_emb_dim: 时间嵌入维度
        hidden_dims: 隐藏层维度列表
    """

    def __init__(self, action_dim: int, obs_dim: int, time_emb_dim: int = 128,
                 hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        input_dim = action_dim + obs_dim + time_emb_dim

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))

        self.net = nn.Sequential(*layers)
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

    def forward(self, a_noisy: torch.Tensor, obs: torch.Tensor,
                k: torch.Tensor) -> torch.Tensor:
        """预测噪声。

        Args:
            a_noisy: (batch, action_dim) 加噪后的动作块
            obs: (batch, obs_dim) 当前观测
            k: (batch,) 扩散时间步 (int tensor)

        Returns:
            noise_pred: (batch, action_dim) 预测的噪声
        """
        t_emb = self.time_emb(k)  # (batch, time_emb_dim)
        x = torch.cat([a_noisy, obs, t_emb], dim=-1)
        return self.net(x)


# ==============================================================================
# 4. Diffusion Policy 完整封装
# ==============================================================================

class DiffusionPolicy:
    """Diffusion Policy — 基于扩散模型的模仿学习策略（x₀ 预测版本）。

    与 ε 预测相比，x₀ 预测直接输出干净动作，模型输出范围受限于数据分布，
    避免 ε→x₀ 转换中的除零放大问题，对低维动作空间更稳定。

    训练:  从专家数据构建 (obs, action_chunk) 对，训练去噪器预测干净动作 x₀
    推理:  对给定 obs，从 N(0,I) 出发用 DDIM 逐步去噪，生成 action_chunk

    Args:
        obs_dim: 观测维度
        act_dim: 单步动作维度
        horizon: 动作块长度 H（一次预测未来 H 步）
        num_diffusion_steps: 训练时的扩散步数 T
        num_inference_steps: 推理时的去噪步数（≤ T，用于加速）
        time_emb_dim: 时间嵌入维度
        denoiser_dims: 去噪器隐藏层维度
        lr: 学习率
        device: 训练设备
    """

    def __init__(self, obs_dim: int, act_dim: int, horizon: int = 8,
                 num_diffusion_steps: int = 100, num_inference_steps: int = 20,
                 time_emb_dim: int = 128, denoiser_dims: list = None,
                 lr: float = 1e-3, device: str = 'cuda'):
        if denoiser_dims is None:
            denoiser_dims = [512, 512, 512]

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = horizon
        self.num_diffusion_steps = num_diffusion_steps
        self.num_inference_steps = num_inference_steps
        self.time_emb_dim = time_emb_dim
        self.denoiser_dims = denoiser_dims
        self.action_chunk_dim = horizon * act_dim

        # 噪声调度器
        self.scheduler = DDPMScheduler(
            num_steps=num_diffusion_steps,
            schedule='cosine',
        ).to(self.device)

        # 去噪器（输出直接是干净动作 x₀，而非噪声 ε）
        self.denoiser = ActionChunkDenoiser(
            action_dim=self.action_chunk_dim,
            obs_dim=obs_dim,
            time_emb_dim=time_emb_dim,
            hidden_dims=denoiser_dims,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # 标准化参数
        self.obs_mean = None
        self.obs_std = None
        self.act_mean = None
        self.act_std = None

        n_params = sum(p.numel() for p in self.denoiser.parameters())
        print(f"[DP] 设备: {self.device}")
        print(f"[DP] 预测目标: x₀ (干净动作), 动作块 horizon={horizon}, 展平={self.action_chunk_dim}")
        print(f"[DP] 扩散步数: 训练 T={num_diffusion_steps}, 推理 T_infer={num_inference_steps}")
        print(f"[DP] 去噪器: {obs_dim}→{time_emb_dim}+{denoiser_dims}, 参数量={n_params:,}")

    # --------------------------------------------------------------------------
    # 数据预处理
    # --------------------------------------------------------------------------

    def _build_chunks(self, observations: np.ndarray, actions: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
        N = len(observations)
        n_chunks = N - self.horizon + 1
        act_chunks = np.zeros((n_chunks, self.action_chunk_dim), dtype=np.float32)
        for i in range(n_chunks):
            act_chunks[i] = actions[i:i + self.horizon].flatten()
        obs_chunks = observations[:n_chunks]
        print(f"[DP] 数据块: {N} 帧 → {n_chunks} 个 (obs, {self.horizon}-step chunk)")
        return obs_chunks, act_chunks

    def _normalize(self, obs: np.ndarray, act_chunks: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
        self.obs_mean = obs.mean(axis=0)
        self.obs_std = obs.std(axis=0) + 1e-8
        self.act_mean = act_chunks.mean(axis=0)
        self.act_std = act_chunks.std(axis=0) + 1e-8
        obs_norm = (obs - self.obs_mean) / self.obs_std
        act_norm = (act_chunks - self.act_mean) / self.act_std
        return obs_norm, act_norm

    # --------------------------------------------------------------------------
    # 训练 (x₀ 预测)
    # --------------------------------------------------------------------------

    def train(self, observations: np.ndarray, actions: np.ndarray,
              epochs: int = 300, batch_size: int = 64, val_split: float = 0.2,
              patience: int = 50):
        """训练 Diffusion Policy（x₀ 预测范式）。

        对每个 (obs, clean_x0) 样本:
          1. 采样 k ~ Uniform(0, T-1), ε ~ N(0, I)
          2. 加噪: a_noisy = √ᾱ_k · x_0 + √(1-ᾱ_k) · ε
          3. 预测: x̂_0 = denoiser(a_noisy, obs, k)
          4. Loss = MSE(x̂_0, x_0)

        Returns:
            history: dict with 'train_loss' and 'val_loss' lists
        """
        obs_chunks, act_chunks = self._build_chunks(observations, actions)
        obs_norm, act_norm = self._normalize(obs_chunks, act_chunks)

        obs_tensor = torch.FloatTensor(obs_norm).to(self.device)
        act_tensor = torch.FloatTensor(act_norm).to(self.device)

        n = len(obs_tensor)
        n_val = int(n * val_split)
        indices = torch.randperm(n)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        print(f"[DP] 数据: {n} 个样本, 训练 {n - n_val}, 验证 {n_val}")
        print(f"[DP] Early stop patience: {patience} epochs")

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_epoch = 0
        no_improve = 0
        T = self.num_diffusion_steps

        for epoch in range(epochs):
            # ---- 训练 ----
            self.denoiser.train()
            perm = torch.randperm(len(train_idx))
            epoch_loss = 0.0
            batches = 0

            for i in range(0, len(train_idx), batch_size):
                batch_idx = perm[i:i + batch_size]
                idx = train_idx[batch_idx]

                obs_batch = obs_tensor[idx]    # (B, obs_dim)
                x0_batch = act_tensor[idx]     # (B, action_chunk_dim) — 干净动作
                B = obs_batch.shape[0]

                # 1. 随机噪声和时间步
                noise = torch.randn(B, self.action_chunk_dim, device=self.device)
                k = torch.randint(0, T, (B,), device=self.device)

                # 2. 加噪
                a_noisy = self.scheduler.add_noise(x0_batch, noise, k)

                # 3. 预测 x₀（而非噪声）
                x0_pred = self.denoiser(a_noisy, obs_batch, k)

                # 4. 损失: MSE(x̂₀, x₀)
                loss = self.loss_fn(x0_pred, x0_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            train_loss = epoch_loss / max(batches, 1)

            # ---- 验证 ----
            self.denoiser.eval()
            with torch.no_grad():
                B_val = len(val_idx)
                noise_val = torch.randn(B_val, self.action_chunk_dim, device=self.device)
                k_val = torch.randint(0, T, (B_val,), device=self.device)
                a_noisy_val = self.scheduler.add_noise(act_tensor[val_idx], noise_val, k_val)
                x0_pred_val = self.denoiser(a_noisy_val, obs_tensor[val_idx], k_val)
                val_loss = self.loss_fn(x0_pred_val, act_tensor[val_idx]).item()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                no_improve = 0
                self._best_state = {
                    k: v.cpu().clone() for k, v in self.denoiser.state_dict().items()
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

        if hasattr(self, '_best_state'):
            self.denoiser.load_state_dict(
                {k: v.to(self.device) for k, v in self._best_state.items()}
            )

        return history

    # --------------------------------------------------------------------------
    # 推理 (DDIM with x₀ prediction)
    # --------------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, obs: np.ndarray) -> np.ndarray:
        """用 DDIM 采样动作块（x₀ 预测版本）。

        算法:
          a_T ~ N(0, I)
          for k = T-1, ..., 0:
            x̂₀ = denoiser(a_k, obs, k)                           # (1) 预测 x₀
            ε̂ = (a_k - √ᾱ_k · x̂₀) / √(1-ᾱ_k)                    # (2) 反推噪声
            a_{k-1} = √ᾱ_{k-1} · x̂₀ + √(1-ᾱ_{k-1}) · ε̂          # (3) DDIM 更新

        与 ε 预测版本的区别: 步骤 (1) 输出已在数据范围内，步骤 (2) 的除数
        √(1-ᾱ_k) ≥ √(1-α₀) ≈ 0.01，不会除零。

        Args:
            obs: (obs_dim,) 或 (batch, obs_dim) 原始观测

        Returns:
            action_chunk: (batch, horizon, act_dim) 或 (horizon, act_dim) 原始动作块
        """
        self.denoiser.eval()
        single = obs.ndim == 1
        if single:
            obs = obs.reshape(1, -1)

        B = obs.shape[0]
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        obs_norm = (obs_tensor - torch.FloatTensor(self.obs_mean).to(self.device)) / (
            torch.FloatTensor(self.obs_std).to(self.device) + 1e-8)

        # 从纯噪声开始
        a_k = torch.randn(B, self.action_chunk_dim, device=self.device)

        # 构建推理时间步序列
        T_train = self.num_diffusion_steps
        T_infer = self.num_inference_steps

        if T_infer < T_train:
            step_ratio = T_train / T_infer
            infer_steps = [int(T_train - 1 - i * step_ratio) for i in range(T_infer)]
            infer_steps = sorted(set(infer_steps + [0]), reverse=True)
        else:
            infer_steps = list(range(T_train - 1, -1, -1))

        # DDIM 采样循环 (x₀ prediction)
        for k_val in infer_steps:
            k = torch.full((B,), k_val, device=self.device, dtype=torch.long)

            # 预测 x₀
            x0_pred = self.denoiser(a_k, obs_norm, k)

            # 反推噪声: ε̂ = (a_k - √ᾱ_k · x̂₀) / √(1-ᾱ_k)
            alpha_bar_k = self.scheduler.alphas_cumprod[k].unsqueeze(-1)  # (B, 1)
            eps_pred = (a_k - torch.sqrt(alpha_bar_k + 1e-8) * x0_pred) / \
                       torch.sqrt(1.0 - alpha_bar_k + 1e-8)

            if k_val > 0:
                # DDIM 确定性更新
                alpha_bar_prev = self.scheduler.alphas_cumprod[k_val - 1].unsqueeze(-1)
                a_k = torch.sqrt(alpha_bar_prev + 1e-8) * x0_pred + \
                      torch.sqrt(1.0 - alpha_bar_prev + 1e-8) * eps_pred
            else:
                a_k = x0_pred  # 最后一步直接输出 x₀

        # 反标准化
        act_mean_t = torch.FloatTensor(self.act_mean).to(self.device)
        act_std_t = torch.FloatTensor(self.act_std).to(self.device)
        a_k = a_k * act_std_t + act_mean_t

        # reshape → (B, horizon, act_dim)
        a_k = a_k.reshape(B, self.horizon, self.act_dim)

        if single:
            return a_k[0].cpu().numpy()
        return a_k.cpu().numpy()

    # --------------------------------------------------------------------------
    # 模型存取 (兼容 BC 接口)
    # --------------------------------------------------------------------------

    def save(self, path: str):
        """保存模型。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.denoiser.state_dict(),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'horizon': self.horizon,
            'num_diffusion_steps': self.num_diffusion_steps,
            'num_inference_steps': self.num_inference_steps,
            'time_emb_dim': self.time_emb_dim,
            'denoiser_dims': self.denoiser_dims,
            'obs_mean': torch.FloatTensor(self.obs_mean),
            'obs_std': torch.FloatTensor(self.obs_std),
            'act_mean': torch.FloatTensor(self.act_mean),
            'act_std': torch.FloatTensor(self.act_std),
        }, path)
        print(f"[DP] 模型已保存: {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        """从文件加载模型。"""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)

        dp = cls(
            obs_dim=ckpt['obs_dim'],
            act_dim=ckpt['act_dim'],
            horizon=ckpt['horizon'],
            num_diffusion_steps=ckpt['num_diffusion_steps'],
            num_inference_steps=ckpt.get('num_inference_steps', ckpt['num_diffusion_steps']),
            time_emb_dim=ckpt.get('time_emb_dim', 128),
            denoiser_dims=ckpt.get('denoiser_dims', [512, 512, 512]),
            device=device,
        )

        dp.denoiser.load_state_dict(ckpt['model_state_dict'])
        dp.denoiser.to(device)

        dp.obs_mean = ckpt['obs_mean'].numpy()
        dp.obs_std = ckpt['obs_std'].numpy()
        dp.act_mean = ckpt['act_mean'].numpy()
        dp.act_std = ckpt['act_std'].numpy()

        dp.denoiser.eval()
        print(f"[DP] 模型已加载: {path} "
              f"(obs_dim={dp.obs_dim}, act_dim={dp.act_dim}, horizon={dp.horizon})")
        return dp
