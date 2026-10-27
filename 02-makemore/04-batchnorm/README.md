# Lab 04 - Batch Normalization from Scratch

选修：不作为进入 Transformer 的前置条件。课程顺序见 [schedule](../schedule.md)。

## Objective

实现一个同时支持 2-D 和 3-D 输入的 `BatchNorm1d`，明确区分训练期统计量和推理期 running statistics。

## Before Coding

阅读 [slides.md](slides.md)，回答：

1. 训练时 mean/variance 由哪些维度统计？
2. `gamma` 和 `beta` 为什么让网络可以撤销归一化？
3. 为什么 eval 时不能继续使用当前 batch 统计量？
4. Linear 紧接 BatchNorm 时，Linear bias 为什么通常是冗余的？

## Student File

`lab04_batchnorm.py`

实现：

1. `BatchNorm1d.__init__`
2. `BatchNorm1d.forward`
3. `BatchNormMLP.__init__`
4. `BatchNormMLP.forward`

## Staged Checks

```bash
uv run pytest -q 02-makemore/04-batchnorm/check_lab04_batchnorm.py -k construct
uv run pytest -q 02-makemore/04-batchnorm/check_lab04_batchnorm.py -k training
uv run pytest -q 02-makemore/04-batchnorm/check_lab04_batchnorm.py -k eval
uv run pytest -q 02-makemore/04-batchnorm/check_lab04_batchnorm.py -k 'gradient or mlp or three_dimensional'
uv run pytest -q 02-makemore/04-batchnorm/check_lab04_batchnorm.py
```

## Required Experiment

用相同 seed、数据、batch size 和步数比较：

- controlled initialization without BN
- deliberately poor initialization without BN
- deliberately poor initialization with BN

记录 train/dev loss、激活分布和对 batch size 的敏感性。

## Definition of Done

- [ ] 全部 Lab 04 checks 通过。
- [ ] `train()` 与 `eval()` 会改变 forward 使用的统计量。
- [ ] running mean/variance 不接收梯度。
- [ ] 自定义 BN 能插入 MLP 并完成训练。
- [ ] 笔记中区分“BN 计算做了什么”与“BN 为什么有用”。
