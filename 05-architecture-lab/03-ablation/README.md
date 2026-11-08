# Lab 03 · 受控消融

## Objective

用相同数据、tokenizer、训练 token 预算和优化规则，对 baseline/variant 跑至少三个配对 seed，同时报告 dev NLL、速度和内存。

## Start Here

按 [slides](slides.md)：先列出[控制变量](slides.md#intuition)，再完成[配对检查与汇总](slides.md#implementation)，随后估算并运行[实验矩阵](slides.md#training)，最后选读[统计解释边界](slides.md#derivations)。

## Student File

`lab03_ablation.py`

1. `validate_paired_runs`
2. `summarize_runs`

每次运行保存完整 config 和 metrics；汇总函数只负责计算，不会把微小差异自动写成“显著提升”。

## Staged Checks

```bash
uv run pytest -q 05-architecture-lab/03-ablation/check_lab03_ablation.py -k paired_seeds
uv run pytest -q 05-architecture-lab/03-ablation/check_lab03_ablation.py -k summary
uv run pytest -q 05-architecture-lab/03-ablation/check_lab03_ablation.py
```

## Definition of Done

- [ ] baseline/variant 至少三个配对 seed，共享字段经检查。
- [ ] 容量变化有配平或明确披露，预算不足时两边一起缩小。
- [ ] 逐 seed 原始结果、均值、样本标准差和曲线完整。
- [ ] 微基准与端到端速度分开，内存和总计算预算有记录。
- [ ] 配置确定后才 final test；负结果保留。
