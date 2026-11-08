# Lab 02 Notes - Context MLP

## Intuition

1. 一个名字的滑动 context 示例：
2. MLP 比 bigram 多看到了什么：
3. 按 word split 的原因：

## Forward Checkpoints

1. embedding lookup shape:
2. 哪个参数随 context 长度增长：
3. dev/test 的角色：

## Training Checkpoints

1. mini-batch loss 为什么不必单调：
2. embedding lookup 的梯度为什么会累加：
3. 训练和采样必须共用哪些约定：

## Shape Ledger

| Tensor | Formula | Actual shape |
|---|---|---|
| X | `(B, T)` | |
| emb | `(B, T, E)` | |
| embcat | `(B, T*E)` | |
| hpre | `(B, H)` | |
| logits | `(B, V)` | |

## Predictions

- block size 变大后：
- small vs larger model 的 train/dev gap：

## Results

| block size | embedding | hidden | parameters | train NLL | dev NLL |
|---:|---:|---:|---:|---:|---:|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 5 | | | | | |

## Failure Log

| Symptom | Root cause | Evidence | Fix |
|---|---|---|---|
| | | | |

## Explain Back

- embedding 是如何共享统计强度的？
- 当 train/dev 同时高或 gap 过大时分别怎么处理？

## Commit

- checks:
- commit:

