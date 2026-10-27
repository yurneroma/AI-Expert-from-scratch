# Lab 01 · 自动微分

已完成的复习入口。覆盖原 Day 1-3。

## 阅读

- [Karpathy：Building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)，按卡点回看计算图、梯度累加、拓扑排序部分。
- [自己的 Value 实现](../shared/engine.py)；[可视化实现](../shared/visualize.py)。
- 手推照片：[乘加](notes/day1-mul-add.PNG)、[多路径](notes/day1-fanout.PNG)、[tanh](notes/day1-tanh.PNG)。

## 校验

```bash
uv run pytest -q 01-micrograd/01-autograd/test_engine.py
uv run python 01-micrograd/01-autograd/demo_grad_check.py
uv run python 01-micrograd/01-autograd/demo_visualize.py
```

复习问题：为什么梯度用 `+=`？为什么从输出按逆拓扑顺序遍历？中心差分与自动微分分别验证什么？

已有 [计算图](graph.svg)。完整数值梯度覆盖与容差改进可作为后续复习，不要求重做整个 Lab。
