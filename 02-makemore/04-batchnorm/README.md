# Lab 04 - Batch Normalization from Scratch

选修：不作为进入 Transformer 的前置条件。课程顺序见 [schedule](../schedule.md)。

## Objective

实现一个同时支持 2-D 和 3-D 输入的 `BatchNorm1d`，明确区分训练期统计量和推理期 running statistics。

## Before Coding

按 [自学讲义](slides.md)（[浏览器版](slides.html)）边读边做：

1. 先读[直觉](slides.md#intuition)，分清 batch statistics、running statistics 和 gamma/beta。
2. 进入[实现](slides.md#implementation)，完成 TODO 1–4，分别通过 training/eval/3-D/MLP 检查。
3. 进入[训练对照](slides.md#training)，比较有/无 BN，而不是只证明归一化公式能运行。
4. 实验后再看[公式与解释](slides.md#derivations)。

在 `notes.md` 先记录统计轴和 train/eval 预测；gamma/beta 与 bias 冗余的推导不是开始 TODO 的前置条件。

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
