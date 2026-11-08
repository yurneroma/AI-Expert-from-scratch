# Lab 01 Notes - Bigram

## Intuition

1. 起始/结束 token 的作用：
2. bigram 的输入和输出：
3. bigram 因为只看一个字符而学不会什么：

## Count Model Checkpoints

1. uniform NLL 的数值预测与直觉：
2. smoothing 解决的问题：
3. count matrix 的行列语义：

## Neural Model Checkpoints

1. neural weight matrix 的行列语义：
2. 零初始化时的预测与 loss：
3. 训练时和生成时的输入分别来自哪里：

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

