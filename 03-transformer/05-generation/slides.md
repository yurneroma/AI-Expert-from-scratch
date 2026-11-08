---
title: "自回归生成"
lang: zh-CN
---

# 学习路线

[直觉](#intuition) → [实现](#implementation) → [生成对照](#training) → [公式选读](#derivations)。先用固定 dummy model 验证规则，再接训练 checkpoint。

# 第一阶段 · 直觉 {#intuition}

每轮都把当前完整前缀放进模型，只取最后位置 logits：

```text
prompt tokens
-> model -> 最后位置的下一 token 分布
-> 抽一个 token 并追加
-> EOS 或 max_new_tokens 停止
```

训练使用真实目标，生成使用模型自己刚产生的 token。模型参数不变，变的是输入序列。

- greedy 总取最大 logit。
- temperature 改变分布尖锐程度。
- top-k 只保留最高分的 k 个候选。
- 显式 generator 让实验可复现。

# 第二阶段 · 实现 {#implementation}

## TODO 1 · sample_next_token

接受最后维为 vocab 的 logits，返回同前导 shape 的整数 ID。验证 `temperature>0`、`1<=top_k<=V`。top-k 过滤后再 softmax；不要把原始未筛选概率重新引入。

```bash
uv run pytest -q 03-transformer/05-generation/check_lab05_generation.py -k top_k
```

## TODO 2 · generate

encode prompt；空 prompt 只有在给出 `bos_id` 时才合法。循环使用 eval/no_grad，取 `logits[0,-1]`，采样并追加，EOS 停止。`max_new_tokens=0` 应直接返回 prompt。

若设置 `context_length`，每次只送最后几个 token。对 learned absolute positions，这等价于窗口位置重新从 0 开始，是一种明确但会改变长程语义的策略；不能悄悄越界。

返回前恢复模型原 train/eval 状态，异常路径也应尽量用 `try/finally` 保护。

```bash
uv run pytest -q 03-transformer/05-generation/check_lab05_generation.py -k generate_eos
uv run pytest -q 03-transformer/05-generation/check_lab05_generation.py -k empty_prompt
```

# 第三阶段 · 生成对照 {#training}

加载同一个 checkpoint，固定 prompt 和 seed，比较至少三组 temperature。记录完整生成配置，不只保存文本。

相同样本不同不等于模型更好；采样多样性、dev NLL 和模型质量是不同问题。检查生成前后参数逐元素不变，模型模式恢复，重复调用不共享隐藏状态。

# 第四阶段 · 采样公式选读 {#derivations}

temperature 对 logits 缩放：

$$
p_i=\frac{e^{z_i/\tau}}{\sum_je^{z_j/\tau}}.
$$

$\tau<1$ 让高分候选更集中，$\tau>1$ 更平坦。top-k 先选分数最高的集合 $S$，再只在该集合归一化：

$$
p_i=
\begin{cases}
e^{z_i/\tau}/\sum_{j\in S}e^{z_j/\tau},&i\in S,\\
0,&i\notin S.
\end{cases}
$$

因此 top-k=1 总是 greedy（不考虑并列时的固定 tie 规则）。

## 自测

- 为什么只取最后位置 logits？
- 为什么 top-k 后必须重新归一化？
- context 截断对 learned positions 有什么影响？
- 为什么固定 seed 仍要固定模型、prompt 和配置？

## 阅读

实现后看 [CS336 L10](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_10.json) 的 autoregressive generation、prefill/decode 和采样；cache 留到下一课。
