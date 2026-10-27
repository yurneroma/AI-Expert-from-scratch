# 04 · 训练系统：测量与优化

先修：第三课的模型、checkpoint、数据和评价流程已稳定。复用同一个基线，不另起一个训练框架。

预算：W7-8（2026-10-20 至 11-02），32 小时任务 + 8 小时缓冲。目标：能解释瓶颈、验证优化的正确性、报告真实吞吐。当前目录提供阅读和 Lab 规格，脚手架与检查随开课补齐。

| 周 / 预算 | Lab | 先读 | 验收 | 状态 / 完成日期 |
|---|---|---|---|---|
| W7 · 8h | [01 · Profiling](01-profiling/README.md) | [CS336 L2][l2]；[L5 GPU slides][l5] | 参数/激活/优化器内存估算，forward/backward/optimizer 时间分解 | 待做 / — |
| W7-8 · 12h | [02 · 训练效率](02-training-efficiency/README.md) | [CS336 A2][a2] 的 benchmarking、mixed precision 相关规格 | 梯度累积正确性与吞吐；设备支持时做精度对照 | 待做 / — |
| W8 · 12h | [03 · KV Cache](03-kv-cache/README.md) | [CS336 L10][l10] | cache/full-prefix logits 对齐，prefill/decode 测量 | 待做 / — |

## 硬件与范围

CPU/MPS 路径完成正确性、梯度累积和 cache；用实际设备计时，不外推 GPU 结果。CUDA 可用时再增加 autocast、显存统计和 fused attention 对照。单机多进程可帮助理解 DDP，但多卡加速比需要实际多卡测量。

Triton/FlashAttention、DDP/FSDP 是后续选修；本轮的通过条件不依赖先拿到多卡。FP32 基线和关键算子应先测清楚，再用 supported dtype/后端做优化；如果编译或混合精度不支持当前设备，在表中记录条件并完成可运行路径。

每次比较固定模型、输入 shape、batch、序列长度、warmup、同步方式与计时区间；分别报告训练 tokens/s、prefill 延迟、decode ms/token、内存，避免将三者混为一个“速度”。

完成后在各 Lab 的 `notes.md` 记录 before/after 和数值误差，更新本表，进入 [架构实验](../05-architecture-lab/schedule.md)。

[l2]: https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_02.json
[l5]: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf
[l10]: https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_10.json
[a2]: https://github.com/stanford-cs336/assignment2-systems
