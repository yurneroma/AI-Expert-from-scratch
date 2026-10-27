# Lab 02 · 改一个组件

先在 [schedule](../schedule.md) 选择唯一主变更。阅读该组件的一手论文，并把用到的版本/链接记入 `notes.md`；默认问题为 RMSNorm。先推导公式，再动手。

| 选题 | 一手阅读 | 重点 |
|---|---|---|
| RMSNorm | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | 去掉重定中心、RMS 公式、实验条件 |
| RoPE | [RoFormer](https://arxiv.org/abs/2104.09864) | Q/K 旋转、相对位置信息、长度条件 |
| SwiGLU | [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | 门控 FFN、容量配平 |
| GQA | [GQA](https://arxiv.org/abs/2305.13245) | KV heads 分组、容量与解码效率取舍 |

## 默认任务：RMSNorm

1. 对最后一个特征维计算 `x / sqrt(mean(x^2) + eps) * gain`，明确它与减均值的 LayerNorm 的差别。
2. 保持 baseline 的 Pre-LN 位置不变，仅替换 norm；用配置开关保留原路径。
3. 检查 final norm 是否一并替换，并在实验定义中固定这一选择。
4. 训练前做小 tensor 数值检查，再做小 batch 过拟合，最后接回完整训练。

## 验收

- float64 下与独立参考的 forward、输入梯度、gain 梯度对齐；覆盖零输入、小方差、不同 batch/序列长度。
- 开关关闭时 baseline 输出不变；整模型保持因果性，checkpoint 记录结构配置。
- 手工改错归一化维度时测试必须失败，证明测试能检测主要错误。
- 参数量、FLOPs 或内存发生的差异被明确记录。

选择 RoPE 时额外测旋转与 cache offset；选择 GQA 时测 KV 分组/广播；选择 SwiGLU 时先配平中间宽度。代码与检查在进入此 Lab 时按选题补齐。
