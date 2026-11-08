# Lab 03 · Decoder Block 与完整模型

## Objective

组装 Pre-LN residual block 和 TinyGPT：token/position embedding → causal blocks → final norm → vocab head。

## Start Here

按 [slides](slides.md)：先画[数据流](slides.md#intuition)，再做[Block/模型 TODO](slides.md#implementation)，然后跑[过拟合与因果实验](slides.md#training)，最后选读[参数与 LayerNorm 公式](slides.md#derivations)。

## Student File

`lab03_transformer_block.py`

1. `TransformerBlock.forward`
2. `TinyGPT.forward`
3. 已提供 `TransformerConfig` 和 `count_parameters`，先读契约再使用。

## Staged Checks

```bash
uv run pytest -q 03-transformer/03-transformer-block/check_lab03_transformer_block.py -k block
uv run pytest -q 03-transformer/03-transformer-block/check_lab03_transformer_block.py -k model_logits
uv run pytest -q 03-transformer/03-transformer-block/check_lab03_transformer_block.py -k causal
uv run pytest -q 03-transformer/03-transformer-block/check_lab03_transformer_block.py
```

## Definition of Done

- [ ] 固定 seed 重建参数一致，参数计数能手算对账。
- [ ] 梯度到达 embeddings、所有 blocks 和 head。
- [ ] 完整模型保持因果性，B=1/T=1 不丢维。
- [ ] 固定小 batch 能过拟合到明显低于初始 `log(V)`。
- [ ] notes 有结构图、shape、参数表和过拟合曲线。
