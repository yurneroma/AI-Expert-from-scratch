---
title: "单组件改动：RMSNorm"
lang: zh-CN
---

# 学习路线

[变更边界](#intuition) → [数值实现](#implementation) → [接回模型与故障实验](#training) → [公式与其他选题](#derivations)。只改一个组件，先证明局部正确。

# 第一阶段 · 变更边界 {#intuition}

baseline 与 variant 的唯一结构差异应是 norm 类型：

```text
LayerNorm path：减均值、除标准差、gain/bias
RMSNorm path：  不减均值、除 RMS、gain
```

Pre-LN 的位置、attention、FFN、数据和训练预算不变。明确 block 内和 final norm 是否一起替换，并写入 config/checkpoint。

参数少了 bias 是结构差异，必须报告；不能把参数或预算的其他变化同时混进来。

# 第二阶段 · 实现 {#implementation}

## TODO 1 · RMSNorm.forward

对最后特征维计算 mean square，使用 `torch.rsqrt(mean_square + eps)` 缩放，再乘 `gain`。保留所有 batch/time 维；零输入应稳定返回零。

```bash
uv run pytest -q 05-architecture-lab/02-component-change/check_lab02_component_change.py -k forward_and_gradients
uv run pytest -q 05-architecture-lab/02-component-change/check_lab02_component_change.py -k zero
```

公开检查用独立 float64 参考对齐 forward、输入梯度和 gain 梯度。不要只测输出而漏掉梯度。

## TODO 2 · baseline/variant 开关

`build_norm("layer_norm",...)` 返回原 `nn.LayerNorm`；`build_norm("rmsnorm",...)` 返回自定义层；未知字符串给清晰错误。

```bash
uv run pytest -q 05-architecture-lab/02-component-change/check_lab02_component_change.py -k switch
```

# 第三阶段 · 接回模型 {#training}

先在完整 TinyGPT 中切换 norm，检查 B=1/T=1、所有参数梯度和因果性。再做固定小 batch 过拟合，最后才恢复训练。

故意把归一化维度改成 batch 维，确认公开检查或你的新增检查失败，再恢复。记录参数量、微基准和内存变化；微基准变快不等于端到端一定变快。

# 第四阶段 · 公式与其他选题 {#derivations}

RMSNorm：

$$
\operatorname{RMS}(x)=\sqrt{\frac1D\sum_{i=1}^{D}x_i^2+\epsilon},
\qquad
y=\frac{x}{\operatorname{RMS}(x)}\odot gain.
$$

LayerNorm 先减均值，RMSNorm 不重定中心；这一差别是实验要验证的机制，不是预设“一定更好”的结论。

其他选题仍只选一个：

- RoPE：额外检查 Q/K 旋转、cache offset 和长度条件，读 [RoFormer](https://arxiv.org/abs/2104.09864)。
- SwiGLU：先配平中间宽度和容量，读 [GLU Variants](https://arxiv.org/abs/2002.05202)。
- GQA：检查 KV 分组/广播和内存，读 [GQA](https://arxiv.org/abs/2305.13245)。

默认一手阅读：[RMSNorm 论文](https://arxiv.org/abs/1910.07467)。完成数值检查后再读公式和实验条件。
