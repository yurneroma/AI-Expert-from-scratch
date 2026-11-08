# 05 · Transformer 架构实验

复用第三课模型和第四课测量工具。只选一个主组件，完成 `冻结基线 → 可开关改动 → 数值/梯度验证 → 配对消融 → 复现报告`。

| Lab | 产出 | 入口 |
|---|---|---|
| 01 | 冻结 baseline、模块图、可证伪假设 | [任务](01-baseline/README.md) |
| 02 | 默认 RMSNorm，可配置组件替换 | [任务](02-component-change/README.md) |
| 03 | 多 seed 配对实验、均值/离散与资源对照 | [任务](03-ablation/README.md) |
| 04 | 有证据边界的报告与复现命令 | [任务](04-report/README.md) |

```bash
uv run pytest -q 05-architecture-lab/check_scaffold.py
```

默认只改 LayerNorm → RMSNorm。负结果有效，但不能用未对齐的测试、不同 token 预算或单独挑最好 seed 支撑结论。
