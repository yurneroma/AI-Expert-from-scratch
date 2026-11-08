# 04 · 训练系统：测量与优化

先修：第三课的模型、checkpoint、数据和评价流程已稳定。复用同一个基线，不另起一个训练框架。

预算：W7-8（2026-10-20 至 11-02），32 小时任务 + 8 小时缓冲。目标：能解释瓶颈、验证优化的正确性、报告真实吞吐。每个 Lab 都有讲义、TODO、公开检查和实验 notes；先证明正确，再测优化收益。

| 周 / 预算 | Lab | 讲义 | 验收 | 状态 / 完成日期 |
|---|---|---|---|---|
| W7 · 8h | [01 · Profiling](01-profiling/README.md) | [slides](01-profiling/slides.md) | 参数/激活/优化器内存估算，forward/backward/optimizer 时间分解 | 待做 / — |
| W7-8 · 12h | [02 · 训练效率](02-training-efficiency/README.md) | [slides](02-training-efficiency/slides.md) | 梯度累积正确性与吞吐；设备支持时做精度对照 | 待做 / — |
| W8 · 12h | [03 · KV Cache](03-kv-cache/README.md) | [slides](03-kv-cache/slides.md) | cache/full-prefix logits 对齐，prefill/decode 测量 | 待做 / — |

## 硬件与范围

CPU/MPS 路径完成正确性、梯度累积和 cache；用实际设备计时，不外推 GPU 结果。CUDA 可用时再增加 autocast、显存统计和 fused attention 对照。单机多进程可帮助理解 DDP，但多卡加速比需要实际多卡测量。

Triton/FlashAttention、DDP/FSDP 是后续选修；本轮通过条件不依赖多卡。FP32 基线和关键算子先测清楚，再用 supported dtype/后端优化；不支持时在表中记录条件并完成可运行路径。

每次比较固定模型、输入 shape、batch、序列长度、warmup、同步方式与计时区间；分别报告训练 tokens/s、prefill 延迟、decode ms/token、内存。

```bash
uv run pytest -q 04-training-systems/check_scaffold.py
```

完成后在各 Lab `notes.md` 记录 before/after 和数值误差，更新本表，进入 [架构实验](../05-architecture-lab/schedule.md)。
