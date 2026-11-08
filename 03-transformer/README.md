# 03 · 从零训练小型 Transformer

先修：Makemore 必修完成。最终产出是一个自己实现、能训练、生成、保存和恢复的 decoder-only Transformer。

## 统一学习路径

每个 Lab 都按 `直觉与最小例子 → 分段 TODO → 公开检查 → 训练/实验 → 公式选读` 进行。先读当前 Lab 的 README，再打开 slides；不需要先读论文或推完整公式。

| Lab | 产出 | 入口 |
|---|---|---|
| 01 | byte/BPE tokenizer、文档 split、next-token batch | [任务](01-tokenizer/README.md) |
| 02 | causal self-attention 与多头 shape | [任务](02-attention/README.md) |
| 03 | Pre-LN block 与完整 TinyGPT | [任务](03-transformer-block/README.md) |
| 04 | AdamW、LR schedule、验证与训练闭环 | [任务](04-training/README.md) |
| 05 | greedy、temperature/top-k、EOS 生成 | [任务](05-generation/README.md) |
| 06 | checkpoint、RNG、恢复与复现基线 | [任务](06-reproducibility/README.md) |

从仓库根目录检查基础设施：

```bash
uv run pytest -q 03-transformer/check_scaffold.py
```

学生文件保留 TODO；逐 Lab 显式运行对应 `check_*.py`，根目录默认 pytest 不代表课程已完成。实验配置、命令、失败记录写入各目录 `notes.md`，状态只更新 `schedule.md`。
