# Lab 05 - Manual Backpropagation

选修：必修阶段先推导 softmax + CE、affine、embedding 三处梯度；完整 BN backward
可在遇到相关调试问题时回来做。课程顺序见 [schedule](../schedule.md)。

## Objective

对 `embedding -> affine -> batchnorm -> tanh -> affine -> cross entropy` 的整张图手写 backward，不调用 `loss.backward()` 生成你的结果。

autograd 只用作独立验证器。

## Before Coding

阅读 [slides.md](slides.md)，在纸上推导：

1. mean 对每个输入的导数。
2. broadcast addition 的梯度为什么要沿某些轴求和。
3. embedding indexing backward 为什么必须累加重复 index。
4. softmax + cross entropy 合并后 logits 梯度的简洁形式。

## Student File

`lab05_backprop.py`

`forward_loss` 和缓存契约已给出。你需要实现：

1. `manual_backward`
2. `compare_gradients`

按反向图顺序写，每次只解决一个中间张量的梯度。

## Staged Checks

```bash
uv run pytest -q 02-makemore/05-backprop/check_lab05_backprop.py -k forward
uv run pytest -q 02-makemore/05-backprop/check_lab05_backprop.py -k output_layer
uv run pytest -q 02-makemore/05-backprop/check_lab05_backprop.py -k batchnorm
uv run pytest -q 02-makemore/05-backprop/check_lab05_backprop.py -k all_parameters
uv run pytest -q 02-makemore/05-backprop/check_lab05_backprop.py
```

`-k output_layer` 和 `-k batchnorm` 会检查 `manual_backward` 返回字典中的中间梯度，请不要只返回参数梯度。

## Required Experiment

为每个参数记录：

| Parameter | exact match | max abs error | relative error |
|---|---:|---:|---:|
| C | | | |
| W1 | | | |
| b1 | | | |
| bngain | | | |
| bnbias | | | |
| W2 | | | |
| b2 | | | |

然后故意删掉一个 broadcast reduction，记录形状或数值如何失败，再恢复。

## Definition of Done

- [ ] 全部 Lab 05 checks 通过。
- [ ] 全部参数梯度与 autograd 对齐。
- [ ] 能解释至少三处 gradient accumulation/reduction。
- [ ] 能在不看代码的情况下画出整张反向图。
