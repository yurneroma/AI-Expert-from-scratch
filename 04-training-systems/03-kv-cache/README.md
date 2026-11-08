# Lab 03 · KV Cache

## Objective

为每层保存 `(B,H,T,d_head)` K/V，先让 cached decode 与 full-prefix 参考逐位置对齐，再测 prefill/decode 和缓存内存。

## Start Here

按 [slides](slides.md)：先理解[哪些计算可以复用](slides.md#intuition)，再完成[cache 生命周期与单步 attention](slides.md#implementation)，随后做[数值对齐和性能实验](slides.md#training)，最后选读[复杂度与内存公式](slides.md#derivations)。

## Student File

`lab03_kv_cache.py`

1. `KVCache.update`
2. `KVCache.clear`
3. `KVCache.memory_bytes`
4. `cached_attention_step`

## Staged Checks

```bash
uv run pytest -q 04-training-systems/03-kv-cache/check_lab03_kv_cache.py -k appends
uv run pytest -q 04-training-systems/03-kv-cache/check_lab03_kv_cache.py -k maximum
uv run pytest -q 04-training-systems/03-kv-cache/check_lab03_kv_cache.py -k full_prefix
uv run pytest -q 04-training-systems/03-kv-cache/check_lab03_kv_cache.py
```

## Definition of Done

- [ ] 长度 1、不同 prompt、batch 下 cached/full-prefix logits 对齐。
- [ ] 层数、长度、dtype/device 一致，独立生成不共享脏缓存。
- [ ] 超过最大 context 明确拒绝或重算，不静默重置位置。
- [ ] prefill 与 decode 分开计时，cache bytes 随长度记录。
- [ ] notes 解释减少的重复计算、增加的内存和实测收益。
