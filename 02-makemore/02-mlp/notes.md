# Lab 02 Notes - Context MLP

## Pre-lab

1. embedding lookup shape:
2. 按 word split 的原因：
3. embedding lookup 的梯度：
4. dev/test 的角色：

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

