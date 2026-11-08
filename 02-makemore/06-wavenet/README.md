# Lab 06 - WaveNet-Style Hierarchical Context Model

选修：依赖 Lab 04 的自定义 BatchNorm，不作为进入 Transformer 的前置条件。
课程顺序见 [schedule](../schedule.md)。

## Objective

把一次性展平全部 context 的 MLP，改为分层合并相邻字符的网络。核心不是复制完整音频 WaveNet，而是理解层次 receptive field。

```text
8 characters
-> pair adjacent positions: 4 groups
-> pair adjacent groups:     2 groups
-> pair adjacent groups:     1 representation
-> predict next character
```

## Before Coding

按 [自学讲义](slides.md)（[浏览器版](slides.html)）边读边做：

1. 先读[直觉](slides.md#intuition)，手画 8 个字符如何逐层合并。
2. 进入[实现](slides.md#implementation)，依次完成 flatten、block、model 和采样检查。
3. 进入[训练与公平比较](slides.md#training)，同时记录参数量、context、dev NLL 和吞吐。
4. 实验后再看[公式与设计解释](slides.md#derivations)。

先在 `notes.md` 写 shape 和预测；receptive field 与层级数公式不是开始 TODO 的前置条件。

## Student File

`lab06_wavenet.py`

实现：

1. `FlattenConsecutive.forward`
2. `HierarchicalBlock`
3. `WaveNetNameModel`
4. `sample_wavenet`

## Staged Checks

```bash
uv run pytest -q 02-makemore/06-wavenet/check_lab06_wavenet.py -k flatten
uv run pytest -q 02-makemore/06-wavenet/check_lab06_wavenet.py -k block
uv run pytest -q 02-makemore/06-wavenet/check_lab06_wavenet.py -k model
uv run pytest -q 02-makemore/06-wavenet/check_lab06_wavenet.py -k sample
uv run pytest -q 02-makemore/06-wavenet/check_lab06_wavenet.py
```

## Required Experiments

在尽量接近的参数量下比较：

- flat MLP, block size 8
- hierarchical model, block size 8, group size 2
- hierarchical model, block size 16, group size 2

记录参数量、train/dev NLL、每秒训练 step 和固定 seed 样本。

## Definition of Done

- [ ] 全部 Lab 06 checks 通过。
- [ ] 模型接受 `(B, block_size)` 并输出 `(B, vocab_size)`。
- [ ] 所有层次 shape 有一张完整表格。
- [ ] 在 dev set 上与 flat MLP 做公平比较。
- [ ] 能说明这个模型借用了 WaveNet 的什么思想，以及没有复制什么。
