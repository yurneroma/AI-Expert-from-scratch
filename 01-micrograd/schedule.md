# 01 · Micrograd

课程目标：从标量计算图走到一个可训练的 MLP。状态：核心实现已完成（2026-09-08 用户确认）；下表整理已有学习成果，不需要重做。

| Lab | 阅读 / 复习 | 实现与证据 | 状态 |
|---|---|---|---|
| [01 · Autograd](01-autograd/README.md) | [本 Lab 导读](01-autograd/README.md#阅读) | `shared/engine.py`；手推照片、数值梯度与计算图 | 已完成 |
| [02 · MLP 与梯度对齐](02-mlp/README.md) | [MLP Lab 讲义](02-mlp/README.md) | `shared/nn.py`；Neuron/Layer/MLP、PyTorch 对齐 | 已完成 |
| [03 · 训练闭环](03-training/README.md) | [训练 Lab 讲义](03-training/README.md) | `train_moon.py`；loss 曲线、决策边界、准确率测试 | 已完成 |
| [04 · 复盘](04-review/README.md) | 自己的实现与 Karpathy 原版 | 差异记录 / 博客草稿 | 可选，不阻塞下一课 |

共享实现位于 `shared/`；各 Lab 的测试、实验脚本和图像在本 Lab 目录。课程回归：

```bash
uv run pytest -q 01-micrograd
```

既有学习记录：

| 阶段 | 当时的收获（保留原文） |
|---|---|
| 加法、乘法推导 | 计算图为了给程序做数值计算导数用的，和符号推导区别很大。 |
| backward | 每个Value的backward只需要计算自己相对输入的导数，链式法则保证了梯度从最终传递到当前Value 节点 |
| 激活函数 | 激活函数是实现非线性的手段 |
| MLP | 注意MLP的输入和neuron神经元的形状要对得上 |
| two-moons | x1,x2 可以画出平面坐标图， 而y 则是判断颜色边界的依据 |

下一课：[02 · Makemore](../02-makemore/schedule.md)。
