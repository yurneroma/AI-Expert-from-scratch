# Lab 04 · 复现报告与答辩

## Objective

把本课程的假设、代码差异、检查和原始日志整理成有证据边界的报告，并能从干净进程恢复、验证和生成。不另加新论文或新组件。

## Start Here

按 [slides](slides.md)：先做[证据清点](slides.md#intuition)，再完成[报告/复现检查工具](slides.md#implementation)，随后执行[干净进程验收与答辩](slides.md#training)，最后检查[结论边界](slides.md#derivations)。

## Student File

`lab04_report.py`

1. `validate_report_markdown`
2. `validate_reproduction_metadata`

`notes.md` 就是报告模板；工具检查结构完整性，不代替你判断证据是否充分。

## Staged Checks

```bash
uv run pytest -q 05-architecture-lab/04-report/check_lab04_report.py -k complete_report
uv run pytest -q 05-architecture-lab/04-report/check_lab04_report.py -k reproduction_metadata
uv run pytest -q 05-architecture-lab/04-report/check_lab04_report.py
```

## Definition of Done

- [ ] 问题、实现、协议、结果、局限和复现命令完整。
- [ ] 每个主要结论链接到逐次数据、曲线或检查。
- [ ] 干净进程可加载 checkpoint、验证和生成，commit 真实。
- [ ] 能现场切换 baseline/variant、运行关键检查并解释负结果。
- [ ] 博客或公开发布可选，不作为结课前置。
