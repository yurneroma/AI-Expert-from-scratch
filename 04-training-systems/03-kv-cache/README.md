# Lab 03 · KV Cache

阅读：[CS336 L10：Inference](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_10.json)，重点看 prefill、decode、KV memory。

## 任务

1. 为每层保存 K/V，明确 `(B,H,T,d_head)` 的布局与 cache length。
2. prefill 填满 prompt 缓存，decode 每次只追加新 token；绝对位置 offset 与原模型一致。
3. 在本课基线的最大 context 内工作；超长时明确拒绝或重算窗口，不直接裁 cache 后悄悄重置位置。
4. 让生成入口可切换 full-prefix/cache 路径。

## 校验与提交

- eval 且关闭 dropout，同一序列逐步比较两条路径的 logits；覆盖长度 1、不同 prompt 长度和 batch size。
- 多次独立生成不共享脏缓存；cache 的层数、长度、dtype/device 一致。
- 测量 prefill 延迟与固定生成长度的 decode ms/token；记录 cache 内存随序列增长。
- 本目录 `notes.md` 解释减少了哪些重复计算、增加了多少缓存以及实测收益。

状态见 [schedule](../schedule.md)。当前为规格，自动检查需同时覆盖数值对齐和缓存生命周期。
