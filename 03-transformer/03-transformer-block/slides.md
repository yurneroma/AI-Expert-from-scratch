---
title: "Decoder Block 与 TinyGPT"
lang: zh-CN
---

# 学习路线

[直觉](#intuition) → [实现](#implementation) → [小 batch 训练](#training) → [公式与参数](#derivations)。先让数据流跑通，再研究 LayerNorm 或 GELU 细节。

# 第一阶段 · 先画数据流 {#intuition}

一个 Pre-LN block：

```text
x -> LayerNorm -> Causal MHA -> + -> LayerNorm -> GELU FFN -> +
 \----------------------------/    \-----------------------------/
```

两条 residual 加法两侧都必须是 `(B,T,D)`。Pre-LN 表示 norm 在子层之前，原始表示沿 residual 主路继续传递。

完整模型：

```text
token IDs (B,T)
-> token embedding + learned position embedding
-> N 个 TransformerBlock
-> final LayerNorm
-> vocabulary head
-> logits (B,T,V)
```

位置表只有 `context_length` 行，因此 forward 必须拒绝超长输入，不能越界后才报底层错误。

# 第二阶段 · 实现 {#implementation}

## TODO 1 · TransformerBlock

模块已在 `__init__` 注册。forward 只组合两条 Pre-LN 子层，不要在 residual 外重复 norm，也不要丢 attention 的因果约束。

```bash
uv run pytest -q 03-transformer/03-transformer-block/check_lab03_transformer_block.py -k block
```

## TODO 2 · TinyGPT

positions 从 0 到 `T-1`，shape `(T,)`，可广播到 batch。运行所有 blocks、final norm 和 head。检查输入是整数二维 `(B,T)` 且 `T <= context_length`。

```bash
uv run pytest -q 03-transformer/03-transformer-block/check_lab03_transformer_block.py -k model_logits
uv run pytest -q 03-transformer/03-transformer-block/check_lab03_transformer_block.py -k causal
```

固定 seed 创建两个模型应得到相同参数。`count_parameters` 已提供，用来和手算对账。

# 第三阶段 · 先过拟合小 batch {#training}

在正式训练前，用一个固定小 batch 反复训练，关闭随机性。初始均匀分类 loss 约为 `log(V)`；目标是在小样本上明显下降。

若 loss 不降，依次检查：X/Y 移位、causal mask、residual shape、梯度是否到每层、参数是否真正更新。这个实验只验证闭环，不声称泛化。

随后测试完整模型因果性：改变未来 token，过去 logits 不变。记录 B=1、T=1 的结果。

# 第四阶段 · 公式与参数 {#derivations}

LayerNorm 对每个 token 的最后 `D` 维独立归一化，再使用 learned gain/bias。它与 BatchNorm 的 batch/time 统计轴不同。

若 token embedding 为 `V*D`、position 为 `T_max*D`，每个 attention 的 QKV/output 参数和每个 FFN 的两层参数分别按矩阵形状计算；总参数再加上 LayerNorm 和 head。代码计数是验收工具，手算用于发现意外注册或共享。

residual 写为 $x+f(\operatorname{Norm}(x))$，恒等主路让信息和梯度不必每次都经过完整子层。它不保证训练一定稳定，仍要结合 Lab 03 的诊断方法。

## 自测

- 为什么 residual 两侧必须同 shape？
- 为什么 position IDs 每次从 0 开始？
- final norm 和 block 内 norm 各在哪里？
- 为什么小 batch 过拟合能发现断图，却不能证明 dev 会好？

## 阅读

实现后看 [CS336 架构 slides](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) 的 normalization、residual、FFN 和位置部分。
