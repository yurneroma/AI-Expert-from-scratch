# Lab 02 · Causal Self-Attention

阅读：[CS336 架构 slides](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) 中 attention 部分；张量操作不熟时回看 [L2](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_02.json)。

先推导 `QK^T / sqrt(d_head)` 的 shape，说明 softmax 沿哪一维、为什么使用下三角因果 mask。

## 任务

1. 实现单头 scaled dot-product attention，使用稳定 softmax。
2. 拆分/合并多头，实现 Q/K/V 与输出投影，始终保留 batch 维。
3. 明确布尔 mask 或加性 mask 的语义，正确处理 batch size 1 和长度 1。
4. 暂不做 FlashAttention；保存清晰参考路径供后续优化对照。

## 校验与提交

- 关闭 dropout，用 float64 将输出及 Q/K/V 梯度与独立参考计算对齐。
- 改变未来位置的输入，过去位置的输出不变；过去输出对未来输入的梯度为零。
- masked 概率为零，允许位置行和为 1；无 NaN。
- 记录 `(B,T,D) → (B,H,T,d_head) → (B,H,T,T) → (B,T,D)` 的完整 shape 表。
- 提交实现、检查和本目录 `notes.md`。代码及自动检查随开课补齐；进度见 [schedule](../schedule.md)。
