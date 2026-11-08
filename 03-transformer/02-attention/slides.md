---
title: "Causal Self-Attention"
lang: zh-CN
---

# 学习路线

[直觉](#intuition) → [实现](#implementation) → [因果实验](#training) → [公式选读](#derivations)。先用 shape 和数值参考实现，再谈优化。

# 第一阶段 · 直觉 {#intuition}

MLP 的 context 是固定拼接；attention 则让每个位置按相关性读取其他位置的 value。

对当前位置：

1. query 表示“我要找什么”。
2. 每个 key 表示“我这里有什么可匹配”。
3. query 与所有 key 算分数，softmax 变成权重。
4. 加权求和所有 value。

causal 模型在生成第 `t` 个位置时不能看未来，因此位置 `t` 只允许读取 `0..t`。mask 是模型结构的一部分，不是数据增强。

完整 shape：

```text
X       (B,T,D)
Q/K/V   (B,T,D)
split   (B,H,T,d_head)
scores  (B,H,T,T)
output  (B,H,T,d_head) -> merge -> (B,T,D)
```

# 第二阶段 · 实现 {#implementation}

## TODO 1 · 拆/并多头

`D = H * d_head`。拆头后把 head 维放到 time 前；合并时先恢复 `(B,T,H,d_head)`，再 contiguous reshape。用精确值检查 round-trip。

```bash
uv run pytest -q 03-transformer/02-attention/check_lab02_attention.py -k heads
```

## TODO 2 · 单头参考路径

计算 scores 后先应用 mask，再沿最后一个 key 维 softmax，最后乘 V。布尔 mask 约定为 `True=允许`。使用稳定 softmax，masked 位置概率必须为 0；正常行和为 1。

```bash
uv run pytest -q 03-transformer/02-attention/check_lab02_attention.py -k single_attention
```

公开检查用 float64 对齐输出和 Q/K/V 梯度。若失败，先检查 softmax 维度和 scale，再查 mask。

## TODO 3 · 完整 causal MHA

`qkv` 一次输出 `3D`，拆成 Q/K/V，再做 split heads。构造下三角允许 mask，attention 后 merge，并经过 `out_proj`。保留 B/T 维。

```bash
uv run pytest -q 03-transformer/02-attention/check_lab02_attention.py -k causal
```

# 第三阶段 · 因果实验 {#training}

固定模型 eval，把最后一个输入位置改大，前三个位置的输出必须完全不变。再分别测试 `B=1`、`T=1`，并让 loss backward，确认所有投影都有梯度。

在 notes 写 shape ledger 和 mask 的一个小矩阵。后续速度优化必须始终与这条参考路径对齐。

# 第四阶段 · 公式选读 {#derivations}

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_head}}+M\right)V.
$$

$QK^T$ 的最后两维由 `(T,d_head)` 和 `(d_head,T)` 得到 `(T,T)`。除以 $\sqrt{d_head}$ 防止点积随维度增大而尺度膨胀；mask 的加性形式在禁止位置放 $-\infty$。

softmax 沿最后维，因为每一行是“一个 query 对所有允许 key 的分布”。对 batch/head 维 softmax 会混淆不同样本。

## 自测

- 为什么 merge 前需要恢复 token-major 布局？
- 为什么未来输入对过去输出的梯度应为零？
- 为什么 scale 使用 head_dim，而不是 d_model？
- 长度 1 时 causal mask 长什么样？

## 阅读

实现后看 [CS336 架构 slides](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) 的 attention 部分；tensor 操作不熟再看 L2。
