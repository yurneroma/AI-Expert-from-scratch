# Lab 01 · 测出性能基线

## Objective

复用第三课模型，把参数/梯度/优化器/激活/attention scores 的内存拆开，并把 forward/backward/optimizer 时间拆开。

## Start Here

按 [slides](slides.md)：先明确[要测什么](slides.md#intuition)，再完成[三个测量工具](slides.md#implementation)，随后跑[固定协议与 context 趋势](slides.md#training)，最后选读[资源公式](slides.md#derivations)。

## Student File

`lab01_profiling.py`

1. `estimate_training_memory`
2. `benchmark_call`
3. `profile_training_step`

## Staged Checks

```bash
uv run pytest -q 04-training-systems/01-profiling/check_lab01_profiling.py -k memory
uv run pytest -q 04-training-systems/01-profiling/check_lab01_profiling.py -k benchmark
uv run pytest -q 04-training-systems/01-profiling/check_lab01_profiling.py -k training_profile
uv run pytest -q 04-training-systems/01-profiling/check_lab01_profiling.py
```

## Definition of Done

- [ ] 手算和代码估算一致，单位为 bytes，假设写清。
- [ ] warmup、重复次数、同步、median/离散程度完整。
- [ ] context 至少三档，耗时与内存趋势有图。
- [ ] CPU/RAM、CUDA、MPS 不可获得的项目标不可用，不猜数。
- [ ] notes 能支持另一进程重跑，并给出首要瓶颈假设。
