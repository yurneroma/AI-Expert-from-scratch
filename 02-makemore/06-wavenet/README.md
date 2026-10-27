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

阅读 [slides.md](slides.md)，回答：

1. context 长度 8、group size 2 需要多少个层次？
2. `(B, T, C)` 通过 `FlattenConsecutive(2)` 后的 shape 是什么？
3. 为什么 hierarchical model 比一次 flatten 拥有更强的归纳偏置？
4. 参数量和 receptive field 不是同一件事，为什么？

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
