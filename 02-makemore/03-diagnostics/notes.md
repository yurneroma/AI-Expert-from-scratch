# Lab 03 Notes - Diagnostics

## Intuition

1. 为什么 loss 下降仍可能不健康：
2. logits、激活、梯度、更新分别回答什么：

## Tool Checkpoints

1. 初始 loss 的均匀基线：
2. tanh 饱和比例怎样计算：
3. fan-in scaling 的使用规则：
4. update/data ratio 的含义：

## Predictions

| Initialization | Initial loss | Saturation | Gradient flow | Training |
|---|---:|---:|---|---|
| Bad | | | | |
| Controlled | | | | |
| Too small | | | | |

## Results

| Initialization | Initial loss | Saturation | Dev NLL | Median log update/data |
|---|---:|---:|---:|---:|
| Bad | | | | |
| Controlled | | | | |
| Too small | | | | |

Artifacts:

- logits:
- activations:
- gradients:
- updates:

## Failure Log

| Symptom | Hypothesis | Diagnostic evidence | Intervention | Result |
|---|---|---|---|---|
| | | | | |

## Explain Back

- loss 下降但网络仍不健康的例子：
- 初始化改变的是什么训练动力学？

## Commit

- checks:
- commit:

