---
title: "梯度累积与精度对照"
lang: zh-CN
---

# 学习路线

[直觉](#intuition) → [实现](#implementation) → [正确性与性能实验](#training) → [缩放公式](#derivations)。先对齐梯度，再测速度。

# 第一阶段 · 为什么不能直接平均 microbatch loss {#intuition}

两个 microbatch 分别有 2 个和 6 个有效 token。若把它们的 mean NLL 简单平均，两边各占一半，小 batch 的每个 token 权重就会偏大。

梯度累积的目标是模拟一次有效大 batch：按 token 数加权，并在所有 microbatch backward 后才做一次 optimizer 更新。

```text
zero_grad 一次
-> microbatch forward / scaled backward
-> ...
-> 边界处 clip / step
-> 下一个有效大 batch
```

尾部不足一组也是一次有效大 batch，不能丢掉。

# 第二阶段 · 实现 {#implementation}

## TODO 1–2 · 权重与 backward

`microbatch_loss_weights([2,6])` 返回 `[0.25,0.75]`。`accumulate_microbatch_losses` 接收已经 forward 的 mean losses，对每个图乘权重后 backward，让 `.grad` 自然累加；不要在这里 zero 或 step。

```bash
uv run pytest -q 04-training-systems/02-training-efficiency/check_lab02_training_efficiency.py -k loss_weights
uv run pytest -q 04-training-systems/02-training-efficiency/check_lab02_training_efficiency.py -k accumulated_gradient
```

## TODO 3 · step 边界

`num_microbatches=5, accumulation_steps=2` 时，在下标 `1,3,4` 后 step。公开检查覆盖整除与短尾。

```bash
uv run pytest -q 04-training-systems/02-training-efficiency/check_lab02_training_efficiency.py -k boundaries
```

# 第三阶段 · 实验 {#training}

关闭 dropout，在 FP32 小模型比较：一次大 batch 与多次累积的梯度和一次更新。然后固定有效 batch，改变 microbatch 大小，记录吞吐和内存。

设备支持时再增加 autocast：记录 dtype、loss/梯度是否有限、与 FP32 的误差。需要 loss scaling 时写清在哪一步 unscale/clip。编译或 fused attention 分开测编译时间和稳态时间；没变快也保留结果。

# 第四阶段 · 缩放公式 {#derivations}

第 $i$ 个 microbatch 有 $n_i$ 个 token，mean loss 为 $\bar\ell_i$。全局 token-mean loss：

$$
L=\frac{\sum_i n_i\bar\ell_i}{\sum_i n_i}.
$$

因此对每个计算图 backward 的权重是 $n_i/\sum_jn_j$。若每个 microbatch token 数相同，它才退化为常见的 `loss / accumulation_steps`。

梯度裁剪和 optimizer step 必须看到完整累积梯度；在每个 microbatch 内 step 会改变有效 batch 和优化器轨迹。

## 自测

- 为什么不等 token 数不能简单除以 accumulation_steps？
- 为什么 clip 要在 accumulation 边界之后？
- 尾部 microbatch 的分母是什么？
- 性能改善和数值对齐为什么要分别报告？

## 阅读

完成 FP32 对齐后看 [CS336 A2](https://github.com/stanford-cs336/assignment2-systems) 的 benchmarking 和 mixed precision；API 以运行环境的 PyTorch 文档为准。
