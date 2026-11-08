---
title: "BatchNorm 自学讲义"
subtitle: "先跑通 train/eval，再理解归一化为什么有用"
lang: zh-CN
---

# 这堂课

本课实现一个支持 `(B, C)` 和 `(B, T, C)` 的 BatchNorm，并把它插入 MLP。最重要的不是背公式，而是分清：**当前 batch 的统计量、保存下来的 running 统计量、可学习的 scale/offset。**

| 阶段 | 要做什么 | 完成标志 |
|---|---|---|
| [第一阶段：直觉](#intuition) | 看懂特征归一化和 train/eval 差别 | 能说出同一个样本为何随 batch 改变 |
| [第二阶段：实现](#implementation) | 完成 TODO 1–4 | 构造、training、eval、梯度和 MLP checks 通过 |
| [第三阶段：训练对照](#training) | 比较有/无 BN 的初始化实验 | 能用 loss 与激活分布描述差别 |
| [第四阶段：公式与解释](#derivations) | 实验后选读 | 能解释 gamma/beta、bias 冗余和统计约定 |

# 第一阶段 · 先看懂 BatchNorm 在管什么 {#intuition}

## 1. 它整理的是每个特征的分布

Lab 03 看到：预激活过大可能让 tanh 饱和，过小又可能让信号消失。BatchNorm 在每个训练 step 中，对当前 batch 的每个特征分别测量均值和方差，再把它们拉回到相近尺度。

对二维输入 `(B, C)`：跨 batch 统计，每个特征各有一个均值和方差。

对三维输入 `(B, T, C)`：跨 batch 和 time 统计，最后一维 `C` 仍是要保留的特征。

所以它不是“把整个张量变成一个均值和一个方差”，也不是对特征维 `C` 求平均。

## 2. 归一化之后为什么还要 gamma 和 beta

强制每个特征均值 0、方差 1，可能正好不是网络想要的表示。每个特征因此还有可学习的：

- `gamma`：决定归一化后的缩放。
- `beta`：决定归一化后的偏移。

网络可以学习新的尺度和中心，也可以近似撤销归一化。它们与 running statistics 是不同状态：前者通过梯度学习，后者通过训练过程中的统计更新得到。

## 3. 为什么训练和推理必须走两条路

训练时，一个样本的输出会受到同 batch 其他样本影响，因为均值和方差由整个 batch 算出。

推理时不能这样：同一个名字不应该因为碰巧和哪些名字同批而得到不同结果。因此 eval 使用训练期间保存的 running mean/variance，不再读取当前 batch。

```text
train：当前 batch 统计量 -> 归一化；同时更新 running 统计量
eval： running 统计量  -> 归一化；不更新 running 统计量
```

调用 `eval()` 改变的是这个行为，不会自动冻结 gamma/beta。

# 第二阶段 · 按状态与 shape 实现 {#implementation}

## TODO 1 · 注册四类状态：`BatchNorm1d.__init__`

对 `num_features=C`：

| 名称 | Shape | 类型 | 初始值 |
|---|---|---|---|
| `gamma` | `(C,)` | `nn.Parameter` | 1 |
| `beta` | `(C,)` | `nn.Parameter` | 0 |
| `running_mean` | `(C,)` | buffer | 0 |
| `running_var` | `(C,)` | buffer | 1 |

还要保存 `eps` 和 `momentum`。running statistics 要注册为 buffer，随 `state_dict` 保存，但不通过梯度训练。

运行：

```bash
uv run pytest -q 02-makemore/04-batchnorm/check_lab04_batchnorm.py -k construct
```

## TODO 2 · 实现两条 forward 路径：`BatchNorm1d.forward`

### 先确定统计轴

```text
输入 (B, C)：   dim=(0,)
输入 (B, T, C)：dim=(0, 1)
```

使用 `keepdim=True`，统计量就能广播回原输入。不要对最后一维 `C` 归约。

### training 路径

实现顺序：

```text
batch_mean = 当前 batch 的均值
batch_var  = 当前 batch 的总体方差（unbiased=False）
x_hat      = (x - batch_mean) / sqrt(batch_var + eps)
y          = gamma * x_hat + beta
```

同时在 `torch.no_grad()` 中更新 running statistics：

```text
running = (1 - momentum) * running + momentum * batch_stat
```

本 Lab 使用这个 momentum 约定，代码和测试必须一致。running 更新不能进入 autograd 图。

### eval 路径

```text
x_hat = (x - running_mean) / sqrt(running_var + eps)
y     = gamma * x_hat + beta
```

不重新读取 batch 统计量，也不更新 running 状态。

运行：

```bash
uv run pytest -q 02-makemore/04-batchnorm/check_lab04_batchnorm.py -k training
uv run pytest -q 02-makemore/04-batchnorm/check_lab04_batchnorm.py -k eval
uv run pytest -q 02-makemore/04-batchnorm/check_lab04_batchnorm.py -k 'gradient or three_dimensional'
```

检查顺序建议：先看二维训练输出是否每个特征均值 0、总体方差 1，再看 eval 是否完全不改 running 状态，最后看三维广播。

## TODO 3 · 组装深层 MLP：`BatchNormMLP.__init__`

网络结构：

```text
Embedding
-> 重复若干 Linear(no bias) -> BatchNorm1d -> Tanh
-> 最终 Linear head
-> logits
```

第一个 Linear 的输入宽度是 `block_size * embedding_dim`；后续层的输入宽度来自上一层 `hidden_dims`。所有模块要注册到 `nn.Module` 容器中，否则参数不会进入 `model.parameters()`。

隐藏 Linear 在紧接 BN 时通常省略 bias，后面的 `beta` 已能提供偏移。最终 logits 层不接 BN，因为 cross entropy 需要不受限制的类别分数。

运行：

```bash
uv run pytest -q 02-makemore/04-batchnorm/check_lab04_batchnorm.py -k mlp
```

## TODO 4 · 组合 forward：`BatchNormMLP.forward`

shape 主线：

```text
inputs  (B, T)
emb     (B, T, E)
embcat  (B, T*E)
blocks  (B, H_last)
logits  (B, V)
```

保留 batch 维，依次调用已注册的 embedding、blocks 和 head。用一个 batch 做 `logits.sum().backward()`，确认所有参数都有梯度。

运行：

```bash
uv run pytest -q 02-makemore/04-batchnorm/check_lab04_batchnorm.py
```

# 第三阶段 · 训练对照：计算做了什么，不等于证明了为什么有用 {#training}

用相同 seed、数据、batch size、网络结构和训练步数比较：

1. controlled initialization，无 BN。
2. deliberately poor initialization，无 BN。
3. deliberately poor initialization，有 BN。

记录：

- train/dev NLL；
- 各层激活分布和 tanh 饱和比例；
- 梯度与 update/data ratio；
- 不同 batch size 下的敏感性。

先在 `notes.md` 写预测。特别区分两种说法：

- **可直接从计算确认**：training 输出使用 batch 统计，eval 使用 running 统计；归一化后由 gamma/beta 做仿射变换。
- **需要实验或额外证据**：BN 是否让这组初始化训练更稳定、dev 是否改善、batch 变小是否带来更大统计噪声。

BatchNorm 原论文对机制的解释有历史意义，但“为什么有效”仍有后续讨论，不要把论文动机当作本实验已经证明的结论。

## 常见失败

| 现象 | 先查什么 |
|---|---|
| 每个特征没有均值 0、方差 1 | 是否沿错误维度归约，或误用 sample variance |
| eval 输出随 batch 改变 | 是否仍在读取当前 batch 统计 |
| eval 后 running 状态变化 | 更新是否只放在 training 分支 |
| running statistics 出现在梯度里 | 更新是否包在 `no_grad()` |
| 三维输入广播报错 | 统计是否 `keepdim=True`，gamma/beta 是否对齐最后维 |
| 深层 MLP 部分参数没梯度 | 模块是否注册，forward 是否漏用某个 block |

# 第四阶段 · 公式与解释（实验后选读） {#derivations}

## 1. 单个特征的归一化与仿射

对特征 $c$，训练 batch 中所有参与统计的值写成 $x_{c,n}$：

$$
\mu_c=\frac1N\sum_nx_{c,n},
\qquad
\sigma_c^2=\frac1N\sum_n(x_{c,n}-\mu_c)^2.
$$

然后：

$$
\widehat x_{c,n}
=\frac{x_{c,n}-\mu_c}{\sqrt{\sigma_c^2+\varepsilon}},
\qquad
y_{c,n}=\gamma_c\widehat x_{c,n}+\beta_c.
$$

`eps` 防止除零并改善数值稳定性。二维时 $N=B$，三维时 $N=B\,T$。

## 2. gamma/beta 为什么可以近似撤销归一化

若忽略 `eps`，选择：

$$
\gamma_c=\sigma_c,
\qquad
\beta_c=\mu_c,
$$

则 $y_{c,n}\approx x_{c,n}$。这不表示网络总会撤销 BN，只表示归一化没有永久剥夺模型选择其他尺度和偏移的能力。

## 3. 为什么前置 Linear 的 bias 通常冗余

设：

$$
h=xW+b.
$$

对 batch 均值做减法时，常量 $b$ 也出现在均值里：

$$
h-\operatorname{mean}(h)
=xW-\operatorname{mean}(xW).
$$

因此 BN 前的 bias 在中心化时被消去，BN 后的 `beta` 再提供可学习偏移。这是“通常冗余”的结构原因，不是任何网络里都禁止 bias 的绝对规则。

## 4. batch variance 与 running variance 是不同约定

训练归一化使用当前 batch 的总体方差 `unbiased=False`。不同库对 running variance 是否使用无偏校正可能有不同选择。

本 Lab 的目标是按 scaffold 和公开检查保持内部一致，不要把其他库的校正公式混进来。报告结果时写清约定，比只写“用了 BN”更可复现。

## 自测

1. 对 `(B,T,C)`，mean/variance 在哪些轴统计，`keepdim=True` 后 shape 是什么？
2. 为什么同一样本在两个 training batch 中可能有不同输出？
3. 推理时必须保存哪四个 per-feature 张量？
4. `model.eval()` 改变了什么，没有改变什么？
5. 为什么 BN 不是只在整个数据集开始前做一次标准化？

## 阅读

- [Ioffe & Szegedy, Batch Normalization (2015)](https://arxiv.org/abs/1502.03167)
- [PyTorch BatchNorm1d 文档](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)

先完成 train/eval 实现，再用文档核对参数、buffer 和方差约定。
