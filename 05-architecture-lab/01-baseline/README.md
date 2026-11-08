# Lab 01 · 建立架构基线

## Objective

冻结第三课模型、tokenizer、数据、token 预算和评价方式，形成可重跑 baseline，并提出一个可证伪的组件假设。默认后续只改 LayerNorm → RMSNorm。

## Start Here

按 [slides](slides.md)：先画[模块与证据地图](slides.md#intuition)，再完成[manifest 工具](slides.md#implementation)，随后做[干净进程复现](slides.md#training)，最后整理[假设与反证条件](slides.md#derivations)。

## Student File

`lab01_baseline.py`

1. `validate_baseline_manifest`
2. `comparison_mismatches`

## Staged Checks

```bash
uv run pytest -q 05-architecture-lab/01-baseline/check_lab01_baseline.py -k complete_manifest
uv run pytest -q 05-architecture-lab/01-baseline/check_lab01_baseline.py -k comparison
uv run pytest -q 05-architecture-lab/01-baseline/check_lab01_baseline.py
```

## Definition of Done

- [ ] forward 图标明 embedding/positions/norm/attention/FFN/head 的代码和 shape。
- [ ] baseline commit、tokenizer、split、token 预算和 evaluation 冻结。
- [ ] 新进程复现 logits、恢复训练和验证结果。
- [ ] notes 写一个机制、预期效果、副作用和反证条件明确的假设。
