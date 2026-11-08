---
title: "可复现基线"
lang: zh-CN
---

# 学习路线

[状态直觉](#intuition) → [checkpoint 实现](#implementation) → [中断恢复实验](#training) → [复现契约](#derivations)。先明确要恢复什么，再写序列化代码。

# 第一阶段 · 只保存权重为什么不够 {#intuition}

加载权重可以重新 forward，却不等于能从同一点继续训练。还需要：

```text
step / data_position
optimizer moments
scheduler 或 LR 位置
Python / torch RNG
tokenizer 与 merge 顺序
模型结构配置
```

缺任何一项，都可能出现“logits 一样，但下一步训练不同”。配置兼容也必须在加载前检查，不能把不匹配的 state_dict 底层报错当作完整契约。

# 第二阶段 · 实现 {#implementation}

## TODO 1 · RNG

保存 Python `random` 和 torch CPU RNG 的独立快照。恢复后下一组随机数应逐项一致；保存时不能留下会被后续修改的引用。

```bash
uv run pytest -q 03-transformer/06-reproducibility/check_lab06_reproducibility.py -k rng
```

## TODO 2–3 · checkpoint

保存模型/可选 optimizer/scheduler、step、data position、config、tokenizer config 和 RNG。加载时先比结构配置，再加载状态，按参数决定是否恢复 RNG。

公开检查覆盖 logits、optimizer state、训练位置和错误配置。设备映射、CUDA RNG 可作为实际训练的扩展，并在 notes 写明。

```bash
uv run pytest -q 03-transformer/06-reproducibility/check_lab06_reproducibility.py -k checkpoint_restores
uv run pytest -q 03-transformer/06-reproducibility/check_lab06_reproducibility.py -k incompatible
```

# 第三阶段 · 中断恢复实验 {#training}

用一个确定性 CPU 小模型：

1. 跑固定 batch 序列，在中途保存。
2. 连续运行若干步，记录下一 batch、LR、loss 和参数。
3. 新进程加载 checkpoint，重跑相同后续步。
4. 在约定容差内比较。

再执行三条干净进程命令：训练恢复、验证、生成。记录真实 commit 和环境，不把“当前工作区”误写成可复现版本。

# 第四阶段 · 复现契约 {#derivations}

复现报告至少回答：谁、在哪台机器、用哪个 commit、哪份数据/tokenizer、哪个配置和 seed、运行什么命令、得到什么原始 artifact。

“随机 seed 固定”只覆盖随机数来源之一。数据迭代位置、优化器状态和结构配置不匹配，仍然无法恢复同一训练轨迹。

最小 CPU 配置用于验证正确性，日常配置用于实验；两者应分开记录，避免把功能 smoke test 当成性能基线。

## 自测

- 为什么 logits 一致还不能证明恢复训练一致？
- 为什么配置应先于 state_dict 加载检查？
- 为什么 tokenizer merge 顺序也属于 checkpoint 契约？
- 哪些结论需要在新进程中验证？

## 阅读

回看自己的 Lab 01–05，再对照 [CS336 A1](https://github.com/stanford-cs336/assignment1-basics) 的完整训练任务，列出覆盖与未覆盖部分。
