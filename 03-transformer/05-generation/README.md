# Lab 05 · 自回归生成

阅读：[CS336 L10：Inference](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_10.json)，先看自回归生成、prefill/decode 和采样概念；缓存实现留到下一课程。

## 任务

1. 完整前缀 forward 后取最后位置 logits，支持 greedy 与 temperature/top-k sampling。
2. 检查 temperature、top-k 和最大长度；支持 EOS；采样使用显式随机种子。
3. 使用 eval/no_grad，明确超出最大 context 时如何截断和重置绝对位置。
4. 输入 prompt → 编码 → 生成 → 解码输出，CLI 接收 checkpoint 与生成参数。

## 校验与提交

- 固定模型、prompt、seed、配置得到相同结果；top-k=1 与 greedy 一致。
- EOS 停止、最大长度、空 prompt 的约定均可验证；不越过位置表索引。
- 生成不改变模型参数，退出后恢复原来的 train/eval 状态。
- 本目录 `notes.md` 保存三组相同 prompt 不同 temperature 的输出，并解释采样变化与模型质量的区别。

状态见 [schedule](../schedule.md)。当前为作业规格。
