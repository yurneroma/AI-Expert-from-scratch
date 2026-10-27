# Lab 04 Notes - BatchNorm

## Pre-lab

1. 2-D / 3-D 的统计维度：
2. gamma/beta 的作用：
3. eval 为什么使用 running stats：
4. Linear bias 为什么冗余：

## State Contract

| State | Shape | Parameter / buffer | Train behavior | Eval behavior |
|---|---|---|---|---|
| gamma | | | | |
| beta | | | | |
| running_mean | | | | |
| running_var | | | | |

## Predictions

- poor init without BN:
- poor init with BN:
- batch size 变小：

## Results

| Variant | Batch size | Train NLL | Dev NLL | Notes |
|---|---:|---:|---:|---|
| controlled, no BN | | | | |
| poor init, no BN | | | | |
| poor init, BN | | | | |

## Failure Log

| Symptom | Root cause | Evidence | Fix |
|---|---|---|---|
| | | | |

## Explain Back

- BN 在数学上做了什么？
- 它为什么可能改善训练？哪部分是你的实验证据？

## Commit

- checks:
- commit:

