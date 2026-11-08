---
title: "KV Cache"
lang: zh-CN
---

# 学习路线

[复用直觉](#intuition) → [cache 实现](#implementation) → [对齐与测量](#training) → [公式选读](#derivations)。没有数值对齐的速度提升不算通过。

# 第一阶段 · 为什么重复算了 {#intuition}

full-prefix decode 每生成一个 token，都把整个前缀重新经过每层。过去 token 的 K/V 在 eval、参数不变时可以复用，不必每层反复投影。

cache 每层保存：

```text
K/V: (B,H,T,d_head)
```

prefill 一次处理 prompt 并填满 cache；decode 每次只追加新 token，再让新 query 读取所有 cached K/V。绝对位置仍使用原序列 offset。

# 第二阶段 · 实现 {#implementation}

## TODO 1–3 · 生命周期、容量与内存

`update` 沿 dim=2 拼接，校验层号、rank、K/V shape、dtype/device 和最大长度。`clear` 清所有层；`memory_bytes` 使用真实 `element_size()`，K/V 都算。

独立生成必须新建或 clear cache。超长时明确报错，后续模型可选择重算窗口，但不能裁掉 cache 后悄悄把位置归零。

```bash
uv run pytest -q 04-training-systems/03-kv-cache/check_lab03_kv_cache.py -k appends
uv run pytest -q 04-training-systems/03-kv-cache/check_lab03_kv_cache.py -k maximum
```

## TODO 4 · 单步 attention

更新 cache 后，当前 query 对所有已缓存 key 算 scores，沿 cached time softmax，再乘所有 value。当前 query 只看过去和当前，所以这条单步路径不需要额外的下三角 mask。

```bash
uv run pytest -q 04-training-systems/03-kv-cache/check_lab03_kv_cache.py -k full_prefix
```

# 第三阶段 · 对齐与性能 {#training}

在 eval、关闭 dropout 下，对每个生成位置比较 cached 与 full-prefix logits。覆盖 B>1、长度 1 和不同 prompt 长度；每层 cache 长度必须同步。

然后分别报告：

- prefill 总延迟；
- 固定生成长度的 decode ms/token；
- cache bytes 随 T 的增长；
- 数值最大误差。

收益来自减少重复投影和前缀中间层计算，但每步仍有 attention、内存读写和框架开销，不保证所有设备都快。

# 第四阶段 · 公式选读 {#derivations}

每层 cache 元素数：

$$
2BHTd_{head}=2BTD
$$

（K 和 V 两份，且 $H d_{head}=D$）。乘以 dtype bytes，再乘层数，cache 内存随 $T$ 线性增长。

full-prefix 每层都要重算整段表示；cache 用额外内存换过去 K/V 的复用。decode 单步 attention 的 score 数量仍随当前缓存长度增长，因此 cache 不是把所有成本变成常数。

## 自测

- prefill 和 decode 分别做什么？
- 为什么 cache offset 与原绝对位置必须一致？
- 为什么两次生成不能默认共享 cache？
- 为什么 cache 内存随 T 线性，而 full attention scores 随 T²？

## 阅读

完成对齐后看 [CS336 L10](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_10.json) 的 prefill、decode 和 KV memory。
