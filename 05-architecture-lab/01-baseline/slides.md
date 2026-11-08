---
title: "架构基线与假设"
lang: zh-CN
---

# 学习路线

[模块地图](#intuition) → [manifest 实现](#implementation) → [基线复现](#training) → [假设与反证](#derivations)。先冻结对照组，再修改组件。

# 第一阶段 · 基线是一次实验的对照组 {#intuition}

先画一次 forward，标出每个模块的代码位置、输入输出 shape 和资源测量入口。然后明确哪些东西不能随 variant 改变：tokenizer、数据 split、训练 token 数、有效 batch、优化器、LR 规则、evaluation。

“baseline 跑过一次”不等于冻结。需要真实 commit、配置、checkpoint 和可复制命令。

# 第二阶段 · 实现 manifest 检查 {#implementation}

`validate_baseline_manifest` 要求 commit、tokenizer、data split、model config、training tokens、evaluation 和 seed。`comparison_mismatches` 找出会破坏对照的共享字段差异。

这些工具不代替实验判断，只把最明显的不可比配置变成可读错误。

```bash
uv run pytest -q 05-architecture-lab/01-baseline/check_lab01_baseline.py -k complete_manifest
uv run pytest -q 05-architecture-lab/01-baseline/check_lab01_baseline.py -k comparison
```

# 第三阶段 · 干净进程复现 {#training}

从 checkpoint 加载 baseline，比较 logits；再恢复训练一小段，并运行固定 evaluation。确认第四课计时和内存工具可直接复用。

记录参数量、dev NLL、tokens/s、推理延迟和内存。任何重跑差异先成为失败记录，不要急着进入 variant。

# 第四阶段 · 写一个可证伪假设 {#derivations}

默认问题：同一训练预算下，RMSNorm 能否保持 dev NLL，同时改变 norm 算子成本？

写清：

- 机制：去掉重定中心，只按 RMS 缩放。
- 预期：可能减少部分计算，训练表现待测。
- 副作用：表示尺度、训练稳定性和参数差异。
- 反证：配对实验中 dev 明显变差，或微基准收益无法转化为端到端收益。

“会更快、更好”不是充分假设；必须写出什么结果会否定它。

## 阅读

先看自己的第三/第四课代码和报告，再看 [CS336 架构 slides](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf)。
