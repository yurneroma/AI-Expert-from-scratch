# Makemore 实验指南

## 每个 Lab 的标准流程

### 1. Read

- 阅读对应 `slides/` 文件。
- 把所有形状推导手写一遍。
- 在 `notes/labXX.md` 中回答 pre-lab 问题。

### 2. Predict

在跑代码前写下：

- 随机模型的预期 NLL。
- 关键中间张量的形状。
- 你认为最容易出错的一步。

### 3. Implement

- 只修改当前 Lab 指定的 `makemore_lab/labXX_*.py`。
- 按 TODO 编号顺序实现。
- 不在损失图中提前 `.item()` 或转 NumPy。
- 先用小 batch 验证正确性，再扩大训练。

### 4. Check

每份 Lab handout 都给出了从小到大的 `pytest -k ...` 命令。一次只解决一组失败。

### 5. Experiment

每个 Lab 至少做一组控制变量实验。产物放在：

```text
02-makemore/artifacts/labXX/
```

建议保存：

- `metrics.json` 或 Markdown 表格
- 训练/验证 loss 曲线
- 激活或梯度直方图
- 固定 seed 下的生成样本

### 6. Explain

如果不能用两句话解释下面三件事，就还不算完成：

1. 模型在预测什么概率分布？
2. 损失如何衡量这个分布？
3. 本 Lab 的结构改变了模型的哪一种能力？

### 7. Commit

建议 commit 命名：

```text
lab(makemore-01): implement count and neural bigrams
lab(makemore-02): train context MLP language model
experiment(makemore-03): diagnose initialization and saturation
lab(makemore-04): implement batch normalization
lab(makemore-05): derive manual backward pass
lab(makemore-06): build hierarchical context model
docs(makemore): complete model comparison report
```

## 调试顺序

语言模型出错时，固定按以下顺序查：

1. **Data**: 一个单词展开后的 `(context, target)` 是否完全正确？
2. **Shape**: 每个中间张量的 shape 是否与手写推导相同？
3. **Baseline**: 初始 loss 是否接近 `log(vocab_size)`？
4. **Overfit**: 能否把 8-32 个样本的小 batch 训练到极低 loss？
5. **Gradient**: 参数是 leaf tensor 吗？`grad` 存在吗？更新前清零了吗？
6. **Split**: 是否误用 test set 做了调参？
7. **Sampling**: 是否和训练时使用相同的 context 更新规则？

## 成绩不等于理解

课程的验收分为三层：

| 层级 | 证据 |
|---|---|
| Correctness | 公开 checks 通过，形状与数值正确 |
| Understanding | 笔记能解释公式、失败模式与设计取舍 |
| Evidence | 有固定 seed 的可复现实验、图表和样本 |

三层都满足才算完成。

