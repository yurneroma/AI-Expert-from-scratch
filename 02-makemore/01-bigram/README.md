# Lab 01 - Bigram Language Models

## Objective

从零实现两个只看前一个字符的语言模型：

1. 基于计数和平滑的 bigram 模型。
2. 基于 one-hot 输入和单层权重矩阵的神经网络 bigram 模型。

两者应学习同一个条件分布 `P(next_char | current_char)`。

## Before Coding

第一次学习**不需要通读讲义，更不需要先读懂公式**。打开 [Bigram 自学讲义](self-study.md)（[浏览器阅读版](self-study.html)），按下面的顺序边读边做：

1. 只读[第一阶段：直觉](self-study.md#intuition)，理解“看当前字符，预测下一个字符”和边界 `.`。
2. 进入[第二阶段：计数模型](self-study.md#count-model)，每读完一小节就完成对应 TODO 1–5，并运行下方分阶段检查。
3. 进入[第三阶段：神经模型与训练](self-study.md#neural-model)，完成 TODO 6–7，确认 loss 下降后跑完整实验。
4. 代码和训练完成后，再按兴趣阅读[第四阶段：公式推导](self-study.md#derivations)。这部分不是开始写代码的前置条件。

在 `notes.md` 分阶段记录：第一阶段解释边界和 bigram 的局限；计数阶段预测 uniform NLL、解释 smoothing；神经阶段记录权重矩阵的行列语义。不要等所有公式都看懂才动手。

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
