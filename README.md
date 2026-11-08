# 从零训练与修改语言模型

当前起点：手写 micrograd 已完成。现在开始 [Makemore 课表](02-makemore/schedule.md)。

本轮目标：12 周后能独立训练一个小型 Transformer，修改其中一个架构组件，并用数值测试、验证集 loss 和性能数据说明改动的作用。

| 项目 / 课程 | 时间预算 | 主要产出 | 课表 |
|---|---|---|---|
| 01 · Micrograd：自动微分与训练闭环 | 已完成 | 标量 autograd、MLP、PyTorch 对齐、two-moons | [schedule](01-micrograd/schedule.md) |
| 02 · Makemore：语言模型基础 | W1-2 · 09/08-09/21 | Bigram → context MLP；掌握 NLL、张量、数据划分与诊断 | [schedule](02-makemore/schedule.md) |
| 03 · Transformer：从零训练小型 GPT | W3-6 · 09/22-10/19 | Tokenizer、causal attention、完整训练、生成与恢复 | [schedule](03-transformer/schedule.md) |
| 04 · Training Systems：测量与优化 | W7-8 · 10/20-11/02 | 性能基线、梯度累积、精度对照、KV cache | [schedule](04-training-systems/schedule.md) |
| 05 · Architecture Lab：修改与验证 | W9-12 · 11/03-11/30 | 一项架构修改、受控消融、可复现实验报告 | [schedule](05-architecture-lab/schedule.md) |

日期均为 2026 年，按每周约 20 小时专注学习估算。时间不足时顺延课表；每周保留约 20% 时间调试和补漏。先用 CPU/MPS 跑小模型；扩大训练或做 CUDA 专项实验时再确定 GPU 预算。

## 每天只走这一条路径

`README → 项目 schedule.md → 当前 Lab/README.md → slides 直觉 → 分段 TODO/检查 → 训练或实验 → 公式选读 → notes/提交`

不要先通读公式或论文。每个 TODO 都有对应的公开检查；先让最小例子和数值对齐，再扩大训练。

每个项目是一门课程，每个编号子目录是一份作业。例如：

```text
02-makemore/
  schedule.md
  01-bigram/
    README.md              # 任务、分阶段入口与验收
    self-study.md          # 背景/直觉 → 代码 → 实验 → 公式
    lab01_bigram.py         # 保留 TODO 的学生代码
    check_lab01_bigram.py   # 每段对应的公开检查
    run_lab01.py            # 实验入口
    notes.md               # 预测、结果、失败与解释
  02-mlp/
  ...
  shared/                  # 跨 Lab 复用的代码
```

只在项目 `schedule.md` 更新 Lab 状态与完成日期；实验结果写入本 Lab 的 `notes.md`，不再维护每日打卡表。一个 Lab 完成后提交代码、测试和少量结果证据。博客由实验笔记整理，不作为进入下一课的前置条件。

## 本轮取舍

- Makemore 的 Bigram、MLP、训练诊断是必修；BatchNorm、完整手写 tensor backward、WaveNet 保留为选修。
- 第三周开始 Transformer。先训通小模型，再研究组件；124M GPT-2、多卡、FlashAttention、MoE、RLHF 留到后续专题。
- 结课看三件事：能解释数据与梯度流；改动经过独立数值校验；对比实验控制数据、token 预算、超参数与随机种子。改动无收益也可以是有效结论。

02-05 均提供讲义、学生 TODO、公开检查和 notes；03-05 另有浏览器版 slides 与项目级脚手架检查。学生 Lab 检查在 TODO 完成前按预期失败，项目级 `check_scaffold.py` 用于先验证资料和环境。

课程组织参考 [Karpathy Zero to Hero](https://karpathy.ai/zero-to-hero.html) 和 [CS336 2025](https://cs336.stanford.edu/spring2025/)，这是面向当前目标裁剪的自学路线。旧路线图、进度与指南完整保存在 [archive](archive/README.md)。
