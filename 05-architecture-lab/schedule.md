# 05 · Transformer 架构实验

先修：已完成第三课可复现模型与第四课测量工具。目标：选择一个组件，提出假设、实现改动、验证正确性、运行对照实验并写出有边界的结论。

预算：W9-12（2026-11-03 至 11-30），64 小时任务 + 16 小时缓冲。这里的“修改架构”指修改并验证模型内部组件；获得原创性能提升需要实验结果支持。

| 周 / 预算 | Lab | 先读 | 验收 | 状态 / 完成日期 |
|---|---|---|---|---|
| W9 · 12h | [01 · 基线与假设](01-baseline/README.md) | [CS336 架构 slides][l3]；第三课代码/报告 | 模块级结构图、资源表、冻结的 baseline、一个可证伪假设 | 待做 / — |
| W9-10 · 20h | [02 · 组件改动](02-component-change/README.md) | 选定组件的一手论文与 [架构 slides][l3] | 一处可开关的变更、独立 forward/backward 校验 | 待做 / — |
| W11 · 20h | [03 · 受控消融](03-ablation/README.md) | [CS336 L12：Evaluation][l12] | 同数据/token 预算，多 seed 的 loss/速度/内存对照 | 待做 / — |
| W12 · 12h | [04 · 复现与答辩](04-report/README.md) | 自己的代码 diff、实验记录和原始日志 | 可复制命令、结果报告、局限分析、现场解释改动 | 待做 / — |

## 只选一个主问题

| 候选变更 | 需要控制的因素 | 合适的第一条假设 |
|---|---|---|
| LayerNorm → RMSNorm（默认） | 保持 block 的归一化位置、其他结构、训练预算一致；说明参数差异 | 在同一训练预算下能否保持 dev NLL，同时改变 norm 算子成本？ |
| learned positions → RoPE | Q/K 旋转、位置 offset、cache、训练与测试长度 | 标准长度的训练表现如何？位置机制对不同长度有什么影响？ |
| GELU FFN → SwiGLU | 调整中间宽度以接近相同参数/FLOPs；披露残余差异 | 门控 FFN 在相近容量下是否有稳定收益？ |
| MHA → GQA（进阶） | query heads 固定，说明 KV 参数量和容量变化 | KV 内存/解码速度与 NLL 的取舍是什么？ |

默认从 RMSNorm 开始；完成正确性与单项实验后，有剩余时间再做第二项。不能一次引入 RoPE、RMSNorm、SwiGLU 后把收益归因于其中一个。

结课证据：代码能运行；单测能检测错误改动；baseline/variant 都能恢复；至少三个训练 seed（算力不足则缩小两边模型和预算）；报告均值、离散程度、原始曲线和负结果。test 留到配置确定后使用。

本课当前提供作业规格；实验时复用第三课实现，不复制另一套模型。脚手架与检查在选定具体组件后补齐。完成本表即结束本轮 12 周，再根据真正遇到的瓶颈选择 CS336 A2 深入、多卡、数据或后训练。

[l3]: https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf
[l12]: https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_12.json
