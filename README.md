# Summer Camp Project — 具身智能与空间智能

> 浙江大学控制科学与工程学院 工业控制研究所 2026 夏令营  
> 课题：____________________（选定后填写）  
> 导师：____________________（选定后填写）

## 环境配置

```bash
# 1. 创建并激活虚拟环境
conda create -n summer_camp python=3.10 -y
conda activate summer_camp

# 2. 安装依赖
pip install -r requirements.txt
```

## 项目结构

```
├── README.md          ← 本文档
├── requirements.txt   ← Python 依赖
├── .gitignore
├── envs/              ← 自定义 MuJoCo 场景 (XML)
├── algorithms/        ← 算法实现
│   ├── bc.py          ← 行为克隆 (Behavior Cloning)
│   └── diffusion_policy.py ← Diffusion Policy (DDPM + DDIM + x₀预测)
├── scripts/           ← 运行脚本
│   ├── draw_circle.py        ← 第1步: Panda 机械臂画圆
│   ├── collect_pick_place.py  ← 第2步: Pick-and-Place 演示收集
│   ├── collect_demo.py       ← 演示收集 (旧版通用模板)
│   ├── train_bc.py           ← 第3步: BC 训练入口
│   ├── eval_bc.py            ← 第3步: BC 评估
│   ├── train_dp.py           ← 第4步: Diffusion Policy 训练入口
│   ├── eval_dp.py            ← 第4步: DP 评估
│   ├── compare.py            ← 第4步: BC vs DP 对比汇总
│   ├── train.py              ← 训练 (旧版通用模板)
│   └── eval.py               ← 评估 (旧版通用模板)
├── results/           ← 实验输出
│   ├── bc_model.pth      ← BC 模型
│   ├── bc_loss.png       ← BC 训练曲线
│   ├── bc_rollout.mp4    ← BC 评估录屏
│   ├── dp_model.pth      ← Diffusion Policy 模型
│   ├── dp_loss.png       ← DP 训练曲线
│   ├── dp_rollout.mp4    ← DP 评估录屏
│   ├── pick_place_demo.npz ← 专家演示数据
│   ├── pick_place_demo.mp4 ← 专家演示录屏
│   ├── figures/          ← 图表
│   └── logs/             ← 训练日志
└── assets/            ← 额外资源 (Demo 视频等)
```

## 快速开始

```bash
# === 完整链路: 数据 → BC → DP → 对比 ===

# 1. 采集 Pick-and-Place 专家演示数据
python scripts/collect_pick_place.py --no-viewer

# 2. 训练 BC baseline
python scripts/train_bc.py

# 3. 训练 Diffusion Policy
python scripts/train_dp.py

# 4. BC vs DP 对比评估
python scripts/compare.py --episodes 10
```

## 实验结果

| 实验 | 算法 | 配置 | 泛化成功率 | 备注 |
|------|------|------|-----------|------|
| exp1_bc_baseline | BC (MLP) | lr=1e-3, epochs=169 | 0/10 (0%) | 灾难性失败——方块被弹飞 |
| exp2_diffusion_policy | Diffusion Policy (x₀+DDIM) | lr=3e-4, epochs=385 | 2/10 (20%) | 温和失败——方块留在桌面 |

> **核心发现**: DP 的成功率虽然不高，但失败模式与 BC 根本不同——BC 在分布外状态产生失控力弹飞方块，DP 的动作块时间一致性保证轨迹平滑、失败温和。

## 参考资料

- MuJoCo 文档: https://mujoco.readthedocs.io/
- dm_control: https://github.com/google-deepmind/dm_control
- 模型库: `~/mujoco/menagerie/`
- 本地使用指南: `~/mujoco/README.md`
