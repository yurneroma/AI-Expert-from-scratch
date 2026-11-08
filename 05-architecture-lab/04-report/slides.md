---
title: "复现报告与答辩"
lang: zh-CN
---

# 学习路线

[证据清点](#intuition) → [结构检查实现](#implementation) → [复现与答辩](#training) → [结论边界](#derivations)。不新增实验来“凑好结果”，先整理已有证据。

# 第一阶段 · 报告是证据索引 {#intuition}

报告回答六件事：问题与假设、实现证据、实验协议、结果、解释与局限、复现命令。

每个主要 claim 都应能回到原始日志、逐 seed 表、曲线或数值检查。代码 diff 用来解释改动，不代替运行结果；生成样本用于定性观察，不代替 dev/test NLL。

# 第二阶段 · 实现结构检查 {#implementation}

`validate_report_markdown` 检查六个二级标题，并在一个错误里列出缺失项。`validate_reproduction_metadata` 要求 commit、checkpoint、三条命令、环境和至少三个 seed。

```bash
uv run pytest -q 05-architecture-lab/04-report/check_lab04_report.py -k complete_report
uv run pytest -q 05-architecture-lab/04-report/check_lab04_report.py -k reproduction_metadata
```

结构通过不代表内容真实；仍要打开 artifact，核对数字和命令。

# 第三阶段 · 干净进程与现场验收 {#training}

从报告复制命令，在新进程加载 checkpoint、验证、生成。确认 config/tokenizer 匹配，记录 commit 和环境。

不看笔记讲 15 分钟：逐层解释模型，切换 baseline/variant，运行关键正确性检查，解释一次性能测量和一张训练曲线。能够回答“为什么没有提升”，但要指出对应证据。

# 第四阶段 · 结论边界 {#derivations}

把结论分成：

- 直接验证：forward/gradient/shape/因果/cache 等检查。
- 本次实验观察：这些 seed、数据、预算和设备上的结果。
- 尚未证明：其他规模、更长训练、不同数据或硬件上的普遍性。

负结果不需要隐藏。若 dev 接近但速度无收益，或微基准改善没有转化为端到端收益，都是清晰的工程结论。

报告可继续整理成博客或代码审查说明，公开发布由你决定。
