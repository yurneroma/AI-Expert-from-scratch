---
title: "性能基线与 Profiling"
lang: zh-CN
---

# 学习路线

[测量直觉](#intuition) → [工具实现](#implementation) → [基线实验](#training) → [资源公式](#derivations)。先固定测量边界，再讨论优化。

# 第一阶段 · “快”具体指什么 {#intuition}

训练 tokens/s、prefill 延迟、decode ms/token 是三种不同指标。本课先测训练 step：

```text
data / forward / backward / optimizer
```

内存也不能只报参数：训练通常还需要梯度、Adam 的两个状态、为 backward 保存的激活，以及 `(B,H,T,T)` attention scores。

一次计时可能受 warmup、后台任务和异步设备影响。要重复测量，报告中位数和离散程度；加速设备在计时边界同步。

# 第二阶段 · 实现 {#implementation}

## TODO 1 · 内存账本

返回 `parameters`、`gradients`、`optimizer_states`、`activations`、`attention_scores`、`total`，单位全是 bytes。默认 FP32，AdamW 每个参数两个状态。它是显式估算，不假装等于 allocator 峰值。

```bash
uv run pytest -q 04-training-systems/01-profiling/check_lab01_profiling.py -k memory
```

## TODO 2 · 可重复计时

`benchmark_call` 先 warmup，再测 repeats；若传 synchronize，在计时边界调用。返回 mean/median/stdev ms 和 repeats。

```bash
uv run pytest -q 04-training-systems/01-profiling/check_lab01_profiling.py -k benchmark
```

## TODO 3 · 一个真实训练 step

`profile_training_step` 清梯度，分别计时 forward、backward、optimizer.step，并记录 loss。公开检查会确认参数真的更新。

```bash
uv run pytest -q 04-training-systems/01-profiling/check_lab01_profiling.py -k training_profile
```

# 第三阶段 · 基线实验 {#training}

固定模型、batch、context、dtype、seed、warmup 和计时区间。改变至少三档 context，记录耗时、内存估算和可用设备统计。

先写瓶颈假设，再看 profile。CPU/RAM、CUDA 显存和 MPS 统计能力不同；无法测量就标不可用。不要用一次最快结果替代稳定分布。

# 第四阶段 · 资源公式 {#derivations}

FP32 参数数量为 $P$ 时，参数和梯度各占 $4P$ bytes；两个 Adam 状态占 $8P$。attention scores 元素数通常是 $BHT^2$，再乘 dtype 字节数。

这解释了为什么参数只随模型宽度变化，而朴素 attention score 内存随 $T^2$ 增长。估算不包括所有临时 buffer、碎片和框架开销，因此要与设备测量并列，而不是互相冒充。

## 自测

- 为什么 optimizer state 不属于参数量？
- 为什么计时前后可能需要 synchronize？
- 为什么 median 要与离散程度一起报告？
- 哪些内存项目估算可能漏掉？

## 阅读

完成小工具后看 [CS336 L2](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_02.json) 和 [GPU slides](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf)。
