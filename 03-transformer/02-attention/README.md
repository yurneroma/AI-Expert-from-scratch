# Lab 02 · Causal Self-Attention

## Objective

实现清晰、可数值核对的 scaled dot-product attention 和 causal MHA。暂不做 FlashAttention。

## Start Here

按 [slides](slides.md)：先理解[位置之间怎样交换信息](slides.md#intuition)，再完成[分段实现](slides.md#implementation)，随后做[因果实验](slides.md#training)，最后选读[公式](slides.md#derivations)。不需要先推完 `QK^T` 才写代码。

## Student File

`lab02_attention.py`

1. `split_heads` / `merge_heads`
2. `scaled_dot_product_attention`
3. `CausalSelfAttention`

## Staged Checks

```bash
uv run pytest -q 03-transformer/02-attention/check_lab02_attention.py -k heads
uv run pytest -q 03-transformer/02-attention/check_lab02_attention.py -k single_attention
uv run pytest -q 03-transformer/02-attention/check_lab02_attention.py -k causal
uv run pytest -q 03-transformer/02-attention/check_lab02_attention.py
```

## Definition of Done

- [ ] float64 下输出和 Q/K/V 梯度对齐独立参考。
- [ ] 未来输入改变不影响过去输出，masked 概率为 0。
- [ ] batch=1、length=1 不丢维度。
- [ ] notes 有完整 shape 表、mask 语义和检查命令。
