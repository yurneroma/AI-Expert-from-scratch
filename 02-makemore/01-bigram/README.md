# Lab 01 - Bigram Language Models

## Objective

从零实现两个只看前一个字符的语言模型：

1. 基于计数和平滑的 bigram 模型。
2. 基于 one-hot 输入和单层权重矩阵的神经网络 bigram 模型。

两者应学习同一个条件分布 `P(next_char | current_char)`。

## Before Coding

第一次学习先阅读 [Bigram 自学讲义](self-study.md)（[浏览器阅读版](self-study.html)）。讲义包含建模背景、贯穿手算例子、公式推导、自测与可运行核对实验；按第 16 节的对应关系逐步进入下面的 TODO。[slides.md](slides.md) 用于学完后的复习。

读完讲义第一阶段后，在 `notes.md` 回答：

1. 为什么必须把起始和结束事件放进训练数据？
2. 字符均匀分布的平均 NLL 是多少？
3. add-one smoothing 在没见过的 bigram 上解决了什么问题？
4. 计数表与神经网络权重矩阵的行和列各代表什么？

## Student File

`lab01_bigram.py`

按 TODO 1-7 实现：

1. `build_vocab`
2. `build_bigram_counts`
3. `counts_to_probabilities`
4. `average_nll`
5. `sample_bigram`
6. `build_bigram_dataset`
7. `neural_bigram_loss` 和 `train_neural_bigram`

## Staged Checks

```bash
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k vocab
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k 'counts or probabilities'
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k 'nll or sample'
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k 'dataset or neural'
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py
```

## Required Experiment

全部 checks 通过后运行：

```bash
uv run python 02-makemore/01-bigram/run_lab01.py
```

结果会保存到 `artifacts/metrics.json`。

在完整 `names.txt` 上比较：

| Model | Train NLL | Generated samples |
|---|---:|---|
| Uniform baseline | | |
| Count bigram, smoothing=0 | | |
| Count bigram, smoothing=1 | | |
| Neural bigram | | |

再改变 smoothing `0.1, 1, 10`，记录 NLL 和样本变化。

## Definition of Done

- [ ] 全部 Lab 01 checks 通过。
- [ ] 计数模型与神经网络模型都能生成字符串。
- [ ] 能解释 `softmax(W[row])` 为什么对应一行条件概率。
- [ ] `notes.md` 包含实验表和至少一条失败记录。
- [ ] commit 中同时包含代码、检查结果和笔记。
