# 03 · 从零训练小型 Transformer

先修：Makemore 必修完成，能独立解释 embedding → logits → CE → backward → update。
目标：一个自己实现、能够训练、生成、保存和恢复的 decoder-only Transformer。

预算：W3-6（2026-09-22 至 10-19），64 小时任务 + 16 小时缓冲。
每个 Lab 均提供讲义、学生 TODO、公开检查和 notes。按“直觉 → 分段实现 → 训练/实验 → 公式选读”进行，不要先通读论文或推完整公式。

| 周 / 预算 | Lab | 讲义 | 验收 | 状态 / 完成日期 |
|---|---|---|---|---|
| W3 · 8h | [01 · Tokenizer 与数据](01-tokenizer/README.md) | [slides](01-tokenizer/slides.md) | round-trip、训练集学习词表、正确的输入/目标移位 | 待做 / — |
| W3-4 · 12h | [02 · Causal Attention](02-attention/README.md) | [slides](02-attention/slides.md) | shape、参考实现数值/梯度对齐、未来 token 隔离 | 待做 / — |
| W4 · 12h | [03 · Transformer Block](03-transformer-block/README.md) | [slides](03-transformer-block/slides.md) | 完整 logits、参数计数、最小 batch 过拟合 | 待做 / — |
| W5 · 16h | [04 · 训练与验证](04-training/README.md) | [slides](04-training/slides.md) | AdamW、LR schedule、有效 dev loss、训练配置和曲线 | 待做 / — |
| W6 · 6h | [05 · 自回归生成](05-generation/README.md) | [slides](05-generation/slides.md) | temperature/top-k、确定性采样、长度与位置边界 | 待做 / — |
| W6 · 10h | [06 · 可复现基线](06-reproducibility/README.md) | [slides](06-reproducibility/slides.md) | checkpoint 恢复、单命令训练/验证/生成、模型结构图 | 待做 / — |

## 实施范围

先做 character/byte tokenizer 跑通数据，再完成简化 BPE；复杂 Unicode regex 预分词和高速 BPE 是扩展。所有模型对比锁定同一个 tokenizer，避免不同 token 单位下直接比较 NLL。

基线固定为 Pre-LN、learned absolute positions、标准 MHA、GELU FFN；Norm、位置编码和 FFN 保持可替换接口，留给第五课做单变量改动。本课参考 CS336 A1 的任务组织，但不宣称覆盖其完整架构或全部官方测试。

先用约 1M-10M 参数和小语料验证闭环，训练规模根据实测吞吐和可用内存决定。先验收“实现正确 + 可复现下降的 dev loss”，不把大模型规模或样本文采设成第一道门槛。CPU 可做功能测试；长训练再选择 MPS/CUDA。

从仓库根目录先运行：

```bash
uv run pytest -q 03-transformer/check_scaffold.py
```

完成一个 Lab 后，在该目录 `notes.md` 写命令、配置、检查结果和一条解释，提交后更新本表。外部 CS336 阅读链接在各 slides 末尾，用于实现后的对照和补漏。
