# Lab 02 · 梯度累积与精度对照

## Objective

先证明 FP32 梯度累积与一次有效大 batch 对齐，再比较 microbatch/accumulation 的吞吐和内存；设备支持时扩展混合精度。

## Start Here

按 [slides](slides.md)：先看懂[为什么平均方式会改变梯度](slides.md#intuition)，再完成[三个小工具](slides.md#implementation)，然后做[大 batch 对齐与效率实验](slides.md#training)，最后选读[缩放公式](slides.md#derivations)。

## Student File

`lab02_training_efficiency.py`

1. `microbatch_loss_weights`
2. `accumulate_microbatch_losses`
3. `accumulation_step_boundaries`

`accumulate_microbatch_losses` 只 backward 累加，不负责 zero/step；完整训练在边界处再 clip、step、zero。

## Staged Checks

```bash
uv run pytest -q 04-training-systems/02-training-efficiency/check_lab02_training_efficiency.py -k loss_weights
uv run pytest -q 04-training-systems/02-training-efficiency/check_lab02_training_efficiency.py -k accumulated_gradient
uv run pytest -q 04-training-systems/02-training-efficiency/check_lab02_training_efficiency.py -k boundaries
uv run pytest -q 04-training-systems/02-training-efficiency/check_lab02_training_efficiency.py
```

## Definition of Done

- [ ] 不同 token 数的 microbatch 按有效 token 缩放。
- [ ] 尾部不足一组也正确 step；边界外不提前 clip/step。
- [ ] 固定有效 batch，比较不同配置的吞吐与内存。
- [ ] 混合精度不支持时记录限制，FP32 正确性仍完成。
- [ ] notes 保留误差、负结果和硬件条件。
