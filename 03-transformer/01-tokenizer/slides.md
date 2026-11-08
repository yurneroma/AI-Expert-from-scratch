---
title: "Tokenizer 与数据"
lang: zh-CN
---

# 学习路线

[直觉](#intuition) → [TODO 实现](#implementation) → [数据实验](#training) → [规则解释](#derivations)。先做出能 round-trip 的编码器，再学 BPE；不要先读完整篇论文。

# 第一阶段 · 直觉 {#intuition}

语言模型只看到整数。tokenizer 负责：

```text
文本 -> UTF-8 bytes -> 0..255 的整数
文本 -> 学习相邻合并 -> 更长的 token IDs
```

byte 编码已经能覆盖任意 Unicode，包括训练时没见过的新字符。BPE 只是在此基础上合并常见相邻 token，减少序列长度；它不会改变 LM 的下一 token 目标。

tokenizer 训练只看文本并产生 merge 表，不更新语言模型参数。merge 只能从 train 文档学习；dev/test 用来评价未见数据。

文档必须先 split，再分别 tokenize。拼接文档时若添加 BOS/EOS，规则必须固定；不能把文档边界悄悄变成普通字符。

# 第二阶段 · 分段实现 {#implementation}

## TODO 1 · byte round-trip

`byte_encode` 直接使用 UTF-8；`byte_decode` 只接受 0..255，遇到 merged/special ID 要给清晰错误。先通过英文、中文、空白和空字符串。

```bash
uv run pytest -q 03-transformer/01-tokenizer/check_lab01_tokenizer.py -k byte_round
```

## TODO 2 · 完整文档 split

使用局部 seeded RNG，不修改输入，按比例得到 train/dev/test。重复文档若代表同一个原始文档，应整组划分；本 Lab 公开检查先覆盖普通唯一文档。

```bash
uv run pytest -q 03-transformer/01-tokenizer/check_lab01_tokenizer.py -k document_split
```

## TODO 3–4 · BPE 训练与编码

每一轮：

1. 在每份训练文档内部统计相邻 pair。
2. 选择出现次数最多的 pair；并列时按 pair 的整数顺序确定。
3. 新 ID 为 `256 + merge_index`。
4. 非重叠地替换所有出现位置，再进入下一轮。

对 `abab`：先合并 `(97,98) -> 256`，序列成为 `[256,256]`，第二次合并为 257。

encode 按 merge 学习的顺序应用；decode 递归展开 merged ID。BOS/EOS 位于所有 merge ID 之后，decode 时忽略它们。词表大小因此是 `258 + len(merges)`。

```bash
uv run pytest -q 03-transformer/01-tokenizer/check_lab01_tokenizer.py -k bpe_learns
```

## TODO 5 · 保存与加载

JSON 至少保存有序 merges。加载后编码结果和 BOS/EOS ID 必须一致； malformed 结构应明确报错。

```bash
uv run pytest -q 03-transformer/01-tokenizer/check_lab01_tokenizer.py -k save_load
```

## TODO 6 · 下一 token batch

对连续 token 窗口，输入取 `[t:t+T]`，目标取 `[t+1:t+T+1]`。下一窗口从 `t+1` 开始。检查 token 数是否足够，不能用短窗口静默凑 batch。

```text
tokens: 0 1 2 3 4
X:      0 1 2
Y:      1 2 3
```

```bash
uv run pytest -q 03-transformer/01-tokenizer/check_lab01_tokenizer.py -k next_token
```

# 第三阶段 · 数据实验 {#training}

用同一份文档 split 训练 byte-BPE，保存 tokenizer JSON。记录：文档数、merge 数、各 split token 数、平均序列长度、BOS/EOS 规则。

人工打印一个 batch，逐位置确认 `Y[t] == X` 的下一个 token。再比较 byte 与 BPE 的序列长度；不同 tokenizer 下的 NLL 不直接作为模型质量对照。

# 第四阶段 · 规则解释 {#derivations}

BPE 是一种确定性的贪心压缩：每轮选择当前训练语料里最高频的相邻 pair。它不保证得到全局最优分词，也不理解语义。

next-token 目标来自同一个 token 序列的移位，不需要人工标签。若 `X` 包含位置 `t..t+T-1`，`Y` 就包含 `t+1..t+T`；错位一位会让整个训练目标改变。

## 自测

- 为什么 byte 后备能编码 dev 中的新字符？
- 为什么 merge 不能从 dev/test 学习？
- 为什么 decode 必须保留 merge 的层级信息？
- 空 prompt、空文档和文档拼接分别采用什么约定？

## 阅读

完成 round-trip 后再看 [CS336 L1](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_01.json) 和 [A1](https://github.com/stanford-cs336/assignment1-basics) 的 tokenization/data loading。
