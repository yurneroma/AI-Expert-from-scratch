# Lab 04 · 训练与验证

阅读：[CS336 A1 handout](https://github.com/stanford-cs336/assignment1-basics) 中 optimizer、learning rate schedule、checkpoint 与实验部分。先解释 token-mean loss、weight decay 和梯度裁剪的区别。

## 任务

1. 先用 PyTorch AdamW 跑通训练闭环，再在小 tensor 上实现 AdamW 更新并与参考对齐。
2. warmup + cosine decay，裁剪梯度，记录 step、已见 tokens、train/dev NLL、learning rate。
3. 固定数据 split 与 tokenizer；验证采用 eval/no_grad，按有效 token 数加权，单独保留 test。
4. 先跑一次 50-step smoke run，再按吞吐估算有限训练预算，保存配置、曲线和 checkpoint。
5. 对比一次学习率或模型宽度；训练不稳定时先检查 token 移位、mask 和尺度。

## 校验与提交

- 手写 AdamW 的多步更新与参考一致，LR 在 warmup 边界与末步正确。
- 每个 step 梯度更新有限；验证不修改模型参数或优化器状态。
- loss 汇总按有效 token 数加权，改变 evaluation batch size 不应改变 NLL。
- 保存相同数据/配置下可复现的下降曲线，报告 train/dev gap 和固定种子的生成样本。
- 提交实现、检查、配置和本目录 `notes.md` 中的执行命令。状态见 [schedule](../schedule.md)；代码与检查待开课补齐。
