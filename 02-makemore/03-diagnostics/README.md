# Lab 03 - Initialization and Training Diagnostics

## Objective

不再只看 loss，而是通过 logits、激活、梯度和参数更新比率判断训练是否健康。

## Before Coding

按 [自学讲义](slides.md)（[浏览器版](slides.html)）分阶段进行，不需要先读推导：

1. 先读[诊断直觉](slides.md#intuition)，理解 logits、激活、梯度和更新分别回答什么。
2. 进入[实现诊断工具](slides.md#implementation)，依次完成 TODO 1–6 和对应检查。
3. 进入[受控实验](slides.md#training)，先预测三组初始化，再用同一 seed 测量。
4. 实验后按需阅读[公式推导](slides.md#derivations)。

先在 `notes.md` 写预测，再记录测量结果；不要先扫超参数，也不要求读懂方差证明后才开始 TODO 1。

## Student File

`lab03_diagnostics.py`

实现：

1. `recommended_weight_std`
2. `tensor_statistics`
3. `tanh_saturation_fraction`
4. `gradient_statistics`
5. `update_to_data_ratio`
6. `save_distribution_plot`

## Staged Checks

```bash
uv run pytest -q 02-makemore/03-diagnostics/check_lab03_diagnostics.py -k weight_std
uv run pytest -q 02-makemore/03-diagnostics/check_lab03_diagnostics.py -k 'tensor or saturation'
uv run pytest -q 02-makemore/03-diagnostics/check_lab03_diagnostics.py -k 'gradient or update'
uv run pytest -q 02-makemore/03-diagnostics/check_lab03_diagnostics.py
```

## Required Experiment

对同一个 MLP 比较三组初始化：

| Variant | Hidden weight scale | Output weight scale |
|---|---:|---:|
| Bad | `1.0` | `1.0` |
| Controlled | `5/3 / sqrt(fan_in)` | `0.01` |
| Too small | `0.01 / sqrt(fan_in)` | `0.001` |

每组保存：

- step 0 的 logits 分布
- tanh 激活分布和饱和比例
- 训练中期的参数梯度标准差
- 每层 update/data ratio
- train/dev loss 曲线

## Definition of Done

- [ ] 全部 Lab 03 checks 通过。
- [ ] 三组实验只改变初始化。
- [ ] 能根据图判断饱和、梯度过小和更新过大。
- [ ] 在笔记中写出“loss 正常下降但训练仍不健康”的一个例子。

