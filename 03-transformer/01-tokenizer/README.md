# Lab 01 · Tokenizer 与数据

阅读：[CS336 L1：Tokenization](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_01.json)；在 [A1](https://github.com/stanford-cs336/assignment1-basics) 中打开 handout，读 BPE training、encoding 和 data loading 的规格。

先写出文本、UTF-8 bytes、token IDs 三者的关系，再说明 tokenizer 训练和 LM 训练的区别。

## 任务

1. 实现固定 byte 编码/解码，先构造可训练的小数据流。
2. 实现简化 BPE：相邻 pair 计数、确定性的并列选择、merge 表、encode/decode、保存/加载。
3. 先按文档划分 train/dev/test，再用 train 学 merge；每个 split 内独立构造输入和目标。
4. 定义 BOS/EOS 和文档拼接规则，保证 `targets[t] == tokens[t+1]`。

本 Lab 的 BPE 可从整段 UTF-8 bytes 起步；它不是 GPT tokenizer 的完全兼容实现。特殊 token 与 regex 预分词作为扩展。

## 校验与提交

- 英文、中文、空白、空文本的 round-trip；保存/加载结果一致。
- 手算 toy corpus 的前两次 merge，检查编码确实使用学到的 merges。
- split 无原始文档泄漏，词表只从 train 学习；字节词表覆盖 dev 中的新字符。
- 人工检查一个 batch 的 `X/Y`，包括上下文边界与目标移位。
- 在本目录提交实现、检查与 `notes.md`：记录数据来源、划分规则、tokenizer 配置和样例。

状态见 [schedule](../schedule.md)。当前为作业规格，代码与自动检查待本课开课时补齐。
