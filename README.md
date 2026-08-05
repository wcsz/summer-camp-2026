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
│   └── ...
├── scripts/           ← 运行脚本
│   ├── collect_demo.py ← 采集 expert demonstration
│   ├── train.py        ← 训练入口
│   └── eval.py         ← 评估
├── results/           ← 实验输出
│   ├── figures/       ← 图表
│   └── logs/          ← 训练日志
└── assets/            ← 额外资源 (Demo 视频等)
```

## 快速开始

```bash
# 1. 采集 demonstration（生成训练数据）
python scripts/collect_demo.py

# 2. 训练策略
python scripts/train.py

# 3. 评估策略
python scripts/eval.py
```

## 实验结果

| 实验 | 算法 | 配置 | 成功率 | 备注 |
|---|---|---|---|---|
| exp1_bc_baseline | BC | lr=1e-3, epochs=100 | ——% | 待运行 |
| | | | | |

## 参考资料

- MuJoCo 文档: https://mujoco.readthedocs.io/
- dm_control: https://github.com/google-deepmind/dm_control
- 模型库: `~/mujoco/menagerie/`
- 本地使用指南: `~/mujoco/README.md`
