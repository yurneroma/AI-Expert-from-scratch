# Lab 03 - Initialization and Training Diagnostics

## Objective

不再只看 loss，而是通过 logits、激活、梯度和参数更新比率判断训练是否健康。

## Before Coding

阅读 [slides.md](slides.md)，然后预测：

1. 初始 logits 绝对值很大时，初始 cross entropy 为什么高于 `log(V)`？
2. tanh 输入绝对值过大时，哪个导数项接近 0？
3. `std = gain / sqrt(fan_in)` 想保持什么量级？
4. update/data ratio 过小和过大分别意味着什么？

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

