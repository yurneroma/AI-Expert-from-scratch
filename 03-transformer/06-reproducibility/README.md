# Lab 06 · 可复现基线

阅读：自己的 Lab 01-05 实现和配置；对照 [CS336 A1](https://github.com/stanford-cs336/assignment1-basics) 的完整训练任务，列出自己覆盖的部分。

## 任务

1. 保存模型、优化器、scheduler、RNG 状态、数据迭代位置、tokenizer 和配置。
2. 给训练、验证、生成各提供一条可复制命令；数据和 checkpoint 采用可配置路径。
3. 定义最小 CPU 配置和日常训练配置，记录环境版本、硬件、耗时和峰值内存。
4. 在本目录整理一份基线报告，供后面两门课程复用同一个模型和数据。

## 校验与提交

- 在确定性的 CPU 小案例中比较连续训练与中断后恢复：下一 batch、LR、loss、参数在约定容差内一致。
- checkpoint 加载后 logits 一致；损坏或不匹配的配置给出明确错误。
- 从干净进程执行三条命令，能恢复并生成；报告包含真实 commit 标识与检查结果。
- 不看资料讲清一次 forward/backward，并指出第五课可以改动的具体模块。

完成后更新 [schedule](../schedule.md)，进入 [训练系统](../../04-training-systems/schedule.md)。当前为规格，代码与自动检查随进度补齐。
