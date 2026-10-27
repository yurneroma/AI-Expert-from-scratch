# Lab 03 · 受控消融

阅读：[CS336 L12：Evaluation](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_12.json)；回看自己的实验假设。

## 任务

1. baseline 与 variant 使用相同数据、tokenizer、训练 token 数、有效 batch、优化器、LR 规则和评价频率。
2. 使用至少三个配对 seed，记录每次运行的完整配置。涉及容量变化时配平参数量或计算量，不能同时配平时说明选择。
3. 同时报告 dev NLL、训练 tokens/s、推理耗时和内存。速度优化先做微基准，再看端到端影响。
4. 算力有限时缩小两边模型或 token 预算；先按一次短跑估算实验矩阵总成本。
5. 只有配置确定后才做 final test evaluation；任何额外调参都要给 baseline 相近预算。

## 提交

本目录 `notes.md` 包含逐 seed 结果表、均值/标准差、loss 曲线、计时协议、总计算预算和一段解释。三个 seed 只能反映有限波动，不把微小差异写成普遍优势；负结果保留。

状态见 [schedule](../schedule.md)。当前为实验规格。
