---
title: "受控消融"
lang: zh-CN
---

# 学习路线

[控制变量](#intuition) → [配对与汇总工具](#implementation) → [实验矩阵](#training) → [统计边界](#derivations)。先保证可比，再比较均值。

# 第一阶段 · 一次只改组件 {#intuition}

baseline 与 variant 共用：tokenizer、数据 split、训练 token 数、有效 batch、优化器、LR 规则、评价频率。seed 要配对：seed 1 的 baseline 与 seed 1 的 variant 使用对应随机控制。

涉及容量变化时配平参数或计算量；不能同时配平就披露选择。block size、宽度等其他变化不能悄悄混入。

# 第二阶段 · 实现 {#implementation}

`validate_paired_runs` 要求恰好两个 variant、每个至少三个 seed、seed 集合一致、共享 config 字段一致。`summarize_runs` 对每个数值指标报告 mean 和 sample std。

```bash
uv run pytest -q 05-architecture-lab/03-ablation/check_lab03_ablation.py -k paired_seeds
uv run pytest -q 05-architecture-lab/03-ablation/check_lab03_ablation.py -k summary
```

公开检查会故意改变一边的 token budget，要求工具拒绝；汇总必须包含所有 seed，不能只挑最好结果。

# 第三阶段 · 先估成本再运行 {#training}

用一次短跑估算：单次 token 数、吞吐、运行数、总时间。算力不足时同时缩小两边模型或预算，不减少对照的公平性。

逐 seed 保存：dev NLL、训练 tokens/s、推理耗时、内存、配置、artifact 和曲线。先做 norm 微基准，再看端到端；只有配置确定后才做 final test。额外调参时给 baseline 相近预算。

# 第四阶段 · 统计解释边界 {#derivations}

样本均值和样本标准差：

$$
\bar x=\frac1n\sum_i x_i,
\qquad
s=\sqrt{\frac{\sum_i(x_i-\bar x)^2}{n-1}}.
$$

三个 seed 只描述这三次运行的有限波动，不证明普遍优势。微小均值差若远小于波动，应写成不确定；负结果同样是有效结论。

速度与 NLL 是不同目标。可以报告“dev 接近、实测成本改变”，不能仅凭一个指标推出全面更好。

## 自测

- 为什么 seed 要配对而不是各自随机选？
- 为什么 token 预算比训练 step 更能对齐不同吞吐的模型？
- 为什么样本标准差用 n-1？
- 哪些额外调参会破坏公平性？

## 阅读

先看自己的假设和原始日志，再看 [CS336 L12 Evaluation](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_12.json)。
