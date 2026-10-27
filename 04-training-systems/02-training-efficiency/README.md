# Lab 02 · 梯度累积与精度对照

阅读：[CS336 A2](https://github.com/stanford-cs336/assignment2-systems) 的 handout 中性能测量与精度相关部分。具体 PyTorch API 以运行环境对应文档为准。

## 必修路径

1. FP32 下实现梯度累积：按有效 token 数缩放 microbatch loss；一次大 batch 与多次累积比较梯度及一次 optimizer 更新。
2. accumulation 边界后才做 clip、step、zero_grad；验证尾部不足一组 microbatch 的处理。
3. 固定有效 batch，比较不同 microbatch/accumulation 配置的吞吐与内存。

## 设备支持时扩展

使用 autocast 做混合精度对照，记录设备支持的 dtype、loss/梯度有限性与误差；需要时使用 loss scaling。可再测编译或 fused attention，分别记录编译时间和稳态耗时。

通过条件：关闭 dropout 的确定性小模型在 FP32 下更新对齐；报告清楚任何精度变化和硬件限制。优化没变快也要保留结果与原因。

提交实现、检查和本目录 `notes.md`。状态见 [schedule](../schedule.md)；当前为规格。
