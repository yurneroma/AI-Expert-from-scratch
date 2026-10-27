# Lab 01 · 建立架构基线

阅读：[CS336 架构 slides](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf)；第三课训练实现、checkpoint 和第四课性能报告。

## 任务与验收

1. 画出一次 forward，标明 embedding、positions、norm、attention、FFN 和 head 的代码位置及 shape。
2. 冻结 baseline 的代码版本、tokenizer、数据划分、训练 token 预算和 evaluation 方式。
3. 用新的进程复现 baseline 的 logits、恢复训练与验证结果，确认测量工具可以复用。
4. 选择一个主问题：默认将 LayerNorm 换为 RMSNorm。记录预期效果、机制、可能副作用及反证条件。

本目录 `notes.md` 保存基线表和一段实验假设。后续 Lab 复用原模型并通过配置开关切换组件。状态见 [schedule](../schedule.md)。
