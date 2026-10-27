# Lab 02 - Context MLP Language Model

## Objective

复现 Bengio et al. (2003) 的核心结构：用固定长度 context 预测下一字符，并学习字符 embedding。

```text
indices -> embeddings -> concatenate -> Linear -> tanh -> Linear -> logits
```

## Before Coding

阅读 [slides.md](slides.md)，在 `notes.md` 写出：

1. batch size 为 `B`、context 长度为 `T`、embedding 维度为 `C` 时，embedding 查表后的 shape。
2. 为什么 train/dev/test 要按单词分割，不能按展开后的字符样本随机分？
3. 为什么 embedding lookup 也能接收梯度？
4. 为什么不应该根据 test loss 反复调参？

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

