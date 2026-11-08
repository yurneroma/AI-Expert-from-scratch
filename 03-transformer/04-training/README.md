# Lab 04 · 训练与验证

## Objective

把 TinyGPT 接进可检查的训练闭环：warmup+cosine、AdamW、梯度裁剪、token-weighted dev NLL、配置与曲线。

## Start Here

按 [slides](slides.md)：先看懂[训练每一步在改什么](slides.md#intuition)，再完成[四个独立工具](slides.md#implementation)，然后做[50-step smoke run 与对照](slides.md#training)，最后选读[AdamW 推导](slides.md#derivations)。

## Student File

`lab04_training.py`

1. `learning_rate_at_step`
2. `ManualAdamW`
3. `weighted_token_nll`
4. `evaluate_token_nll`

完整训练循环复用 Lab 01–03 的 tokenizer、batch 和 TinyGPT，不另建模型。

## Staged Checks

```bash
uv run pytest -q 03-transformer/04-training/check_lab04_training.py -k learning_rate
uv run pytest -q 03-transformer/04-training/check_lab04_training.py -k adamw
uv run pytest -q 03-transformer/04-training/check_lab04_training.py -k weighted
uv run pytest -q 03-transformer/04-training/check_lab04_training.py -k evaluation
uv run pytest -q 03-transformer/04-training/check_lab04_training.py
```

## Definition of Done

- [ ] AdamW 多步更新与 PyTorch 对齐，LR 边界正确。
- [ ] 验证 eval/no_grad，不改参数，并按有效 token 加权。
- [ ] 改变 evaluation batch size 不改变 NLL。
- [ ] 50-step smoke run 后，再按吞吐安排训练预算。
- [ ] notes 保存配置、命令、train/dev 曲线和一次学习率或宽度对照。
