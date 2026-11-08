# Lab 05 · 自回归生成

## Objective

复用训练好的模型，完成 greedy、temperature/top-k、EOS、最大长度和 context 边界清楚的生成入口。KV cache 留到下一课程。

## Start Here

按 [slides](slides.md)：先理解[完整前缀如何一步步变长](slides.md#intuition)，再完成[采样与生成 TODO](slides.md#implementation)，然后做[同 prompt 对照](slides.md#training)，最后选读[采样分布公式](slides.md#derivations)。

## Student File

`lab05_generation.py`

1. `sample_next_token`
2. `generate`

`generate` 返回 prompt+continuation；空 prompt 必须显式传 `bos_id`。超长时使用最后 `context_length` 个 token，并在 notes 说明绝对位置窗口重置的影响。

## Staged Checks

```bash
uv run pytest -q 03-transformer/05-generation/check_lab05_generation.py -k top_k
uv run pytest -q 03-transformer/05-generation/check_lab05_generation.py -k generate_eos
uv run pytest -q 03-transformer/05-generation/check_lab05_generation.py -k empty_prompt
uv run pytest -q 03-transformer/05-generation/check_lab05_generation.py
```

## Definition of Done

- [ ] top-k=1 与 greedy 一致，同 seed/config 可复现。
- [ ] EOS、最大长度、空 prompt、context 截断均有明确行为。
- [ ] 生成不改参数，退出恢复原 train/eval 状态。
- [ ] notes 保存同 prompt 三组 temperature 输出，并区分采样变化与模型质量。
