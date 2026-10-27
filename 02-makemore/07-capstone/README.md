# Lab 07 - Model Comparison Capstone

## Objective

用约 2 小时整理必修 Lab 01-03 的结果，形成进入 Transformer 前的课程小结。
选修 Lab 04-06 的结果有则补充，不需要为填表额外完成所有选修。

## Required Comparison

使用同一份 train/dev/test split 和固定 evaluation 脚本：

| Model | Parameters | Train NLL | Dev NLL | Test NLL | Training time |
|---|---:|---:|---:|---:|---:|
| Count bigram | | | | | |
| Neural bigram | | | | | |
| MLP | | | | | |
| MLP + BatchNorm（选修） | | | | | |
| Hierarchical model（选修） | | | | | |

## Required Ablation

复用 Lab 02 或 03 的一组实验，只改变一个因素：

- context length
- embedding dimension
- hidden width/depth
- initialization scale
- BatchNorm on/off
- hierarchical grouping depth

至少运行 3 个设置，不允许只报告最好结果。

## Reproducibility Contract

报告必须包含：

- git commit hash
- Python / PyTorch 版本
- device
- random seed
- dataset 来源和 split seed
- 模型配置
- optimizer、learning rate、batch size、steps
- 主要图表的生成命令

## Oral Check

不看笔记，用 10 分钟讲清前 3 题；后三题在完成相关选修后回答：

1. count bigram 到 neural bigram 改变了什么？
2. MLP 为什么能使用更长 context？
3. 初始化为什么会影响训练动力学？
4. BatchNorm 训练和推理的数据依赖有何不同？
5. 手写 backward 让你对 broadcast 和 accumulation 有什么新理解？
6. hierarchical model 如何扩大 receptive field？

## Definition of Done

- [ ] 必修模型对比表完整，选修未做时标记未做。
- [ ] ablation 有明确控制变量。
- [ ] 每个主要结论都有数据或图表支撑。
- [ ] 生成样本使用相同 seed 和数量。
- [ ] 完成 `notes.md` 中的 retrospective。
- [ ] 保存本课程小结；博客可选。
