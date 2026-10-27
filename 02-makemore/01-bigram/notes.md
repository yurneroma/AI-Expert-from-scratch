# Lab 01 Notes - Bigram

## Pre-lab

1. 起始/结束 token 的作用：
2. uniform NLL 推导：
3. smoothing 解决的问题：
4. count matrix 与 neural weight matrix 的行列语义：

## Shape Ledger

| Tensor | Shape | Meaning |
|---|---|---|
| counts | | |
| probabilities | | |
| inputs | | |
| weights | | |
| logits | | |

## Predictions

- uniform NLL:
- smoothing 变大后 NLL/样本会：
- neural bigram 与 count bigram 的 loss 差距：

## Results

| Model / setting | Train NLL | Notes / fixed-seed samples |
|---|---:|---|
| Uniform | | |
| Count, smoothing=0 | | |
| Count, smoothing=0.1 | | |
| Count, smoothing=1 | | |
| Count, smoothing=10 | | |
| Neural bigram | | |

## Failure Log

| Symptom | Root cause | Evidence | Fix |
|---|---|---|---|
| | | | |

## Explain Back

- 为什么 count model 和 neural model 在学同一个分布？
- bigram 模型永远不可能学到什么？

## Commit

- checks:
- commit:

