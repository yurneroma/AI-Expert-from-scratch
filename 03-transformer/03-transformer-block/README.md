# Lab 03 · Decoder Block 与完整模型

阅读：[CS336 架构 slides](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) 中 normalization、residual、FFN 和位置表示。

先画一层 Pre-LN block 的数据流，说明 residual 加法两侧为什么必须有相同 shape。

## 任务

1. token embedding + learned absolute positional embedding。
2. Pre-LN → causal MHA → residual；Pre-LN → GELU FFN → residual。
3. 堆叠 blocks，再接 final LayerNorm 和 vocabulary head，输出 `(B,T,V)`。
4. 用配置控制 `layers/heads/d_model/context_length`，校验维度整除与位置长度。
5. 保持 LayerNorm、位置编码、FFN 接口清楚，便于第五课替换其中一项。

## 校验与提交

- LayerNorm 与 PyTorch 对齐；固定 seed 重建参数一致。
- 参数计数手算与代码一致；梯度到达 embedding、所有 blocks 和输出层。
- 完整模型保持因果性；长度 1、batch size 1 都不丢维度。
- 关闭 dropout，在一个固定小 batch 上过拟合到明显低于初始 `log(V)`，排除断图或错误标签。
- 提交模型、检查和本目录 `notes.md` 中的结构图。进度见 [schedule](../schedule.md)；本目录当前提供规格。
