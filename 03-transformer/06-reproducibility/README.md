# Lab 06 · 可复现基线

## Objective

保存并恢复模型、优化器、scheduler、RNG、数据位置、tokenizer 和结构配置，让连续训练与中断恢复可对齐。

## Start Here

按 [slides](slides.md)：先列出[“接着训练”需要哪些状态](slides.md#intuition)，再完成[RNG/checkpoint TODO](slides.md#implementation)，然后做[中断恢复实验](slides.md#training)，最后整理[复现契约](slides.md#derivations)。

## Student File

`lab06_reproducibility.py`

1. `capture_rng_state` / `restore_rng_state`
2. `save_checkpoint`
3. `load_checkpoint`

## Staged Checks

```bash
uv run pytest -q 03-transformer/06-reproducibility/check_lab06_reproducibility.py -k rng
uv run pytest -q 03-transformer/06-reproducibility/check_lab06_reproducibility.py -k checkpoint_restores
uv run pytest -q 03-transformer/06-reproducibility/check_lab06_reproducibility.py -k incompatible
uv run pytest -q 03-transformer/06-reproducibility/check_lab06_reproducibility.py
```

## Definition of Done

- [ ] CPU 小案例中，连续训练与恢复后的下一 batch、LR、loss、参数对齐。
- [ ] 加载后 logits 一致，配置不匹配给清晰错误。
- [ ] 训练、验证、生成各有一条可复制命令，数据/ckpt 路径可配置。
- [ ] notes 记录 commit、Python/PyTorch、硬件、耗时、内存和 artifacts。
- [ ] 能现场讲清 forward/backward，并指出第五课可替换的模块。
