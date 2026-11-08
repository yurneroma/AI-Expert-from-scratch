---
title: "Makemore 结课对照"
lang: zh-CN
---

# 学习路线

[背景与问题](#intuition) → [结果表检查](#implementation) → [对照与消融](#training) → [解释边界](#derivations)。本课不新增模型或公式，先把已有结果整理成可复核证据。

# 第一阶段 · 为什么不能只挑最好的一次 {#intuition}

前面各 Lab 的 NLL、样本和速度只有在 tokenizer、数据 split、评价边界一致时才能并排。先收齐命令、配置、artifact 和 seed；缺口径就重跑或标不可比。

消融只改变一个因素，并至少保留三个设置。最好结果之外的其他设置用于暴露趋势、代价和负结果。

# 第二阶段 · 结果表检查 {#implementation}

`validate_comparison_rows` 要求 Count bigram、Neural bigram、MLP 三行完整且数值有限。选修未做在报告中标记，不伪造数值。

`validate_ablation_rows` 要求至少三个不同 setting、相同 controlled variables、一个 independent variable 和有限 result。

```bash
uv run pytest -q 02-makemore/07-capstone/check_lab07_capstone.py -k comparison
uv run pytest -q 02-makemore/07-capstone/check_lab07_capstone.py -k ablation
```

# 第三阶段 · 对照与消融 {#training}

先填逐次结果，再写结论。比较参数量、train/dev/test NLL 和时间；生成样本使用相同 seed 和数量。

复用 Lab 02 或 03，选择 context、容量或初始化等单一变量。先写控制变量和预测，再跑至少三个设置，并保留所有结果。

# 第四阶段 · 解释边界 {#derivations}

train NLL 描述拟合，dev 用于选择，test 只做最后报告。样本看起来更好不等于 dev 一定更好，一次运行也不证明稳定优势。

结课口头解释重点：count 到 neural 改了什么，MLP 如何利用 context，初始化怎样改变训练动力学。选修题只在完成对应 Lab 后回答。
