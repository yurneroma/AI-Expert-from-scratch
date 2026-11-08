---
title: "训练与验证"
lang: zh-CN
---

# 学习路线

[直觉](#intuition) → [独立工具](#implementation) → [smoke run 与实验](#training) → [优化器公式](#derivations)。先把每个工具单独测准，再组装训练循环。

# 第一阶段 · 直觉 {#intuition}

一个 training step：

```text
取 next-token batch
-> forward / token-mean cross entropy
-> zero_grad / backward
-> clip gradients
-> AdamW step
-> 记录 step、tokens、LR、train loss
```

验证是另一条路：eval/no_grad，不更新参数或优化器。dev 用来选设置，test 留到最后。

三个容易混淆的对象：

- token-mean loss：每个有效目标 token 等权。
- gradient clipping：限制反向得到的梯度范数。
- weight decay：优化器更新中的参数衰减，不等同于把 L2 项随意加进报告 NLL。

# 第二阶段 · 实现 {#implementation}

## TODO 1 · warmup + cosine

step 从 0 开始。warmup 阶段用 `(step+1)/warmup_steps`，所以 warmup 最后一步达到 base LR；之后 cosine 在 `total_steps-1` 到 min LR。拒绝越界 step 和非法总步数。

```bash
uv run pytest -q 03-transformer/04-training/check_lab04_training.py -k learning_rate
```

## TODO 2 · ManualAdamW

状态在每个参数第一次使用时已有 `m/v` 零张量。每一步读取 `.grad`，更新一阶/二阶矩，做 bias correction，再应用 decoupled weight decay。全程 no_grad，多个 step 与 PyTorch AdamW 对齐。

```bash
uv run pytest -q 03-transformer/04-training/check_lab04_training.py -k adamw
```

## TODO 3–4 · token-weighted 验证

若每个 evaluation batch 的 loss 是 sum NLL，直接相加；若记录的是 mean NLL，必须先乘回该 batch token 数。最终只除以有效 token 总数，不能对 batch mean 再简单平均。

`evaluate_token_nll` 接收 `(X,Y)` batches，模型输出 `(B,T,V)`。合并 logits/targets 的 batch/time 维，用 `reduction=sum`，并在返回前恢复模型原 train/eval 状态。

```bash
uv run pytest -q 03-transformer/04-training/check_lab04_training.py -k weighted
uv run pytest -q 03-transformer/04-training/check_lab04_training.py -k evaluation
```

# 第三阶段 · 训练与实验 {#training}

先跑 50-step smoke run：确认 loss 有限且下降、每步梯度有限、checkpoint 可保存。再测量吞吐，估算训练预算。

训练循环至少记录：

```text
step / seen_tokens / LR / train NLL / dev NLL / elapsed
```

固定 tokenizer 和数据 split，对比一次学习率或模型宽度。报告 train/dev gap；训练不稳定时先查 token 移位、mask、初始尺度和 LR，不要立即扩大模型。

# 第四阶段 · AdamW 公式选读 {#derivations}

对梯度 $g_t$：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

偏差校正：

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t}.
$$

AdamW 更新：

$$
\theta_t=\theta_{t-1}-\eta_t
\left(\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}
+\lambda\theta_{t-1}\right).
$$

weight decay 直接衰减参数，与梯度 $g_t$ 分开；这就是 decoupled 的含义。cosine decay 只改变 $\eta_t$，不改变 Adam 的状态更新。

## 自测

- 为什么 warmup step 0 不直接使用完整 LR？
- 为什么 evaluation batch 的 mean loss 不能直接平均？
- clip、weight decay、NLL 各影响哪一步？
- 为什么 50-step smoke run 应先于长训练？

## 阅读

完成工具后看 [CS336 A1](https://github.com/stanford-cs336/assignment1-basics) 的 optimizer、schedule、training loop 和 experiments。
