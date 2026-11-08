# Lab 01 · Tokenizer 与数据

## Objective

实现 UTF-8 byte 后备编码、简化 byte-BPE、文档级 split 和下一 token batch。它不是 GPT tokenizer 的兼容实现；复杂预分词留作扩展。

## Start Here

按 [slides](slides.md) 的四阶段走，不需要先读论文：

1. [直觉](slides.md#intuition)：分清文本、bytes、token IDs，以及 tokenizer 训练与 LM 训练。
2. [实现](slides.md#implementation)：TODO 1–6，每段立即跑检查。
3. [数据实验](slides.md#training)：只用 train 学 merge，人工核对一个 X/Y batch。
4. [表示与规则解释](slides.md#derivations)：实验后回看 BPE 和移位规则。

## Student File

`lab01_tokenizer.py`

- `byte_encode` / `byte_decode`
- `split_documents`
- `ByteBPETokenizer.train / encode / decode / save / load`
- `build_next_token_batch`

## Staged Checks

```bash
uv run pytest -q 03-transformer/01-tokenizer/check_lab01_tokenizer.py -k byte_round
uv run pytest -q 03-transformer/01-tokenizer/check_lab01_tokenizer.py -k document_split
uv run pytest -q 03-transformer/01-tokenizer/check_lab01_tokenizer.py -k bpe_learns
uv run pytest -q 03-transformer/01-tokenizer/check_lab01_tokenizer.py -k save_load
uv run pytest -q 03-transformer/01-tokenizer/check_lab01_tokenizer.py -k next_token
uv run pytest -q 03-transformer/01-tokenizer/check_lab01_tokenizer.py
```

## Definition of Done

- [ ] 英文、中文、空白、空文本 round-trip。
- [ ] 能手算 toy corpus 的两次 merge，保存/加载一致。
- [ ] split 无文档泄漏，merge 只从 train 学习。
- [ ] `targets[t] == tokens[t+1]`，边界与特殊 token 规则写入 notes。
- [ ] notes 包含数据来源、配置、检查命令和样例。
