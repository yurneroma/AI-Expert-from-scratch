# Lab 02 - Context MLP Language Model

## Objective

复现 Bengio et al. (2003) 的核心结构：用固定长度 context 预测下一字符，并学习字符 embedding。

```text
indices -> embeddings -> concatenate -> Linear -> tanh -> Linear -> logits
```

## Before Coding

不需要先通读 [自学讲义](slides.md)（[浏览器版](slides.html)），按阶段边读边做：

1. 先读[直觉](slides.md#intuition)，手画一个名字的滑动 context。
2. 进入[实现](slides.md#implementation)，依次完成 TODO 1–4 和对应检查。
3. 进入[训练与实验](slides.md#training)，完成 TODO 5–6，再比较 context 和容量。
4. 实验后按需阅读[公式与解释](slides.md#derivations)，不作为写代码的前置条件。

在 `notes.md` 分阶段记录 word split 的理由、forward shape 和 train/dev 预测，不要等到全部理论读完才动手。

## Student File

`lab02_mlp.py`

按 TODO 1-6 实现：

1. `split_words`
2. `build_context_dataset`
3. `initialize_parameters`
4. `mlp_logits` 和 `evaluate_mlp_loss`
5. `train_mlp`
6. `sample_mlp`

## Staged Checks

```bash
uv run pytest -q 02-makemore/02-mlp/check_lab02_mlp.py -k split
uv run pytest -q 02-makemore/02-mlp/check_lab02_mlp.py -k dataset
uv run pytest -q 02-makemore/02-mlp/check_lab02_mlp.py -k 'initialize or logits'
uv run pytest -q 02-makemore/02-mlp/check_lab02_mlp.py -k 'train or sample'
uv run pytest -q 02-makemore/02-mlp/check_lab02_mlp.py
```

## Required Experiments

### Experiment A: Context length

固定其他设置，比较 `block_size = 1, 2, 3, 5`。

### Experiment B: Capacity

比较两组结构：

- small: embedding 2, hidden 50
- larger: embedding 10, hidden 200

记录 train/dev loss，判断欠拟合还是过拟合。

## Definition of Done

- [ ] 全部 Lab 02 checks 通过。
- [ ] train/dev/test 分割可复现且无单词泄漏。
- [ ] dev NLL 明显好于 bigram baseline。
- [ ] 绘制 train/dev loss 曲线。
- [ ] 用形状表解释一次 forward pass。

