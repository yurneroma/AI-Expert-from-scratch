---
title: "分层 Context / WaveNet 自学讲义"
subtitle: "先用形状看懂层级，再比较结构与训练结果"
lang: zh-CN
---

# 这堂课

本课把 Lab 02 的一次性展平 context，改成反复合并相邻位置。核心不是复制完整音频 WaveNet，而是理解：**怎样用分层结构和参数共享，让表示看到越来越长的上下文。**

| 阶段 | 要做什么 | 完成标志 |
|---|---|---|
| [第一阶段：直觉](#intuition) | 看懂相邻位置逐层合并 | 能手画 8 个字符的三层树 |
| [第二阶段：实现](#implementation) | 完成 TODO 1–4 | flatten、block、model、sample checks 通过 |
| [第三阶段：公平比较](#training) | 训练 flat 与 hierarchical 模型 | 同口径记录参数、dev NLL 和吞吐 |
| [第四阶段：公式与设计解释](#derivations) | 实验后选读 | 能解释 receptive field、层级数和共享 |

# 第一阶段 · 把长 context 看成一棵树 {#intuition}

## 1. flat MLP 的瓶颈

Lab 02 对 `T` 个字符、`E` 维 embedding 一次性拼接：

```text
(B, T, E) -> (B, T*E) -> Linear(T*E, H)
```

context 变长时，第一层输入宽度随之增长，而且所有绝对位置在一个大变换里混合。它没有显式表达“相邻字符先形成局部模式，再组合成更长模式”。

## 2. 每两个相邻位置合并一次

以 8 个字符为例：

```text
8 个字符
-> 两两合并：4 个局部表示
-> 两两合并：2 个更长表示
-> 两两合并：1 个完整 context 表示
-> 预测下一个字符
```

shape 变化是：

```text
(B, 8, E)
-> (B, 4, H)
-> (B, 2, H)
-> (B, 1, H)
-> (B, V)
```

第一层每个表示看到 2 个字符，第二层看到 4 个，第三层看到 8 个。这就是 receptive field 随层数扩大。

## 3. 同一层为什么共享变换

同一个 level 的所有相邻组使用同一份 Linear/BN 参数。这样局部模式不必为每个绝对位置各学一套变换。

这是结构上的归纳偏置：它鼓励模型复用局部规律。是否比 flat MLP 更好，仍要在参数量、训练步数和 dev NLL 接近可比的条件下测量。

## 4. 它和完整 WaveNet 的关系

本课借用：

- 自回归的下一 token 预测；
- 分层处理局部上下文；
- 随深度指数增长的 receptive field。

原始 WaveNet 用 causal、dilated convolutions 处理音频，并包含更完整的 gated/residual 结构。本 Lab 是字符模型，不是完整音频架构复刻。

# 第二阶段 · 用 shape 驱动实现 {#implementation}

## TODO 1 · 相邻位置拼接：`FlattenConsecutive.forward`

输入输出约定：

```text
输入： (B, T, C)
输出： (B, T/group_size, C*group_size)
```

以 group size 2 为例，只重排时间维，并把相邻两组的 channel 按原顺序拼接：

```text
[位置0的通道, 位置1的通道] -> 一个新位置的 2C 通道
```

实现前检查：

- 输入必须是三维。
- `T` 必须能被 `group_size` 整除。
- reshape 不能改变字符先后和 channel 顺序。
- batch 维始终保留。

运行：

```bash
uv run pytest -q 02-makemore/06-wavenet/check_lab06_wavenet.py -k flatten
```

公开检查用 `(1,4,3)` 验证 `(1,2,6)` 和精确元素顺序；另一个检查要求不兼容的时间维抛出 `ValueError`。

## TODO 2 · 一个分层 block：`HierarchicalBlock`

组合顺序固定为：

```text
FlattenConsecutive(group_size)
-> Linear(group_size * input_channels, output_channels, bias=False)
-> BatchNorm1d(output_channels)
-> Tanh
```

Linear 的输入宽度必须使用**拼接后的 channel 数**，不是原来的 `input_channels`。

对 `(B,8,3)`、group size 2、输出宽度 7：

```text
(B, 8, 3) -> (B, 4, 6) -> (B, 4, 7)
```

运行：

```bash
uv run pytest -q 02-makemore/06-wavenet/check_lab06_wavenet.py -k block
```

## TODO 3 · 组装完整模型：`WaveNetNameModel`

从 embedding 开始，重复 block，直到时间维变成 1：

```text
inputs    (B, block_size)
embedding (B, block_size, E)
blocks    (B, 1, H)
head      (B, V)
```

构造时：

- 保存 `self.block_size`。
- 注册 embedding、`nn.ModuleList` 形式的 blocks 和最终 head。
- 根据 group size 验证能否形成完整二叉/多叉层级；基础 Lab 不接受不能整除到底的 block size。
- 第一层输入 channel 是 embedding 宽度，后续层使用上一层的输出宽度。

forward 最后只移除 singleton time 维。不要使用不带维度的 `squeeze()`：当 `B=1` 时，它会把 batch 维也删掉。

运行：

```bash
uv run pytest -q 02-makemore/06-wavenet/check_lab06_wavenet.py -k model
```

## TODO 4 · 生成循环不变：`sample_wavenet`

模型结构变了，但从 context 到下一字符的接口没变：

```text
从 block_size 个边界 token 开始
-> 取当前 context，增加 batch 维
-> model.eval() 且 no_grad 下得到 logits
-> softmax 抽样
-> 边界则停止，否则左移并追加
-> 达到 max_length 停止
```

因为模型含 BatchNorm，采样必须使用 eval；函数返回前恢复模型原来的 train/eval 状态。生成多个名字时沿用同一 generator，不要每个名字重置 seed。

运行：

```bash
uv run pytest -q 02-makemore/06-wavenet/check_lab06_wavenet.py -k sample
uv run pytest -q 02-makemore/06-wavenet/check_lab06_wavenet.py
```

# 第三阶段 · 训练与公平比较 {#training}

比较：

1. flat MLP，`block_size=8`。
2. hierarchical model，`block_size=8, group_size=2`。
3. hierarchical model，`block_size=16, group_size=2`。

尽量让第一组对照的参数量接近，并记录：

| 项目 | 为什么必须记录 |
|---|---|
| 参数量 | 防止把更大模型的收益归给层级结构 |
| context 长度 | block size 16 同时增加了输入信息 |
| train/dev NLL | 区分拟合与泛化 |
| steps/s 或 wall-clock | 层级结构也有计算代价 |
| steps、batch size、seed | 保证训练预算和可复现性 |
| 固定 seed 样本 | 只作定性观察，不替代 dev NLL |

先在 `notes.md` 写预测，再填结果。block size 8 与 16 的比较不是纯结构 ablation，因为 context 也变了；结论中要明确这一点。

## 常见失败

| 现象 | 先查什么 |
|---|---|
| reshape 后元素顺序变了 | 是否按 `(B,T,C)` 合并相邻时间位置 |
| Linear 输入宽度错误 | 是否使用 `group_size * input_channels` |
| B=1 时输出 shape 错误 | 是否用了 unrestricted `squeeze()` |
| 构造 block size 6 的模型 | 基础 Lab 要求能按 group size 整除到底 |
| 采样结果随 batch 异常变化 | 是否切到 eval 并使用 running statistics |
| 调用采样后模型一直停在 eval | 是否恢复调用前的 training 状态 |
| 声称层级一定更好 | 是否控制参数量、训练预算和 context |

# 第四阶段 · 公式与设计解释（实验后选读） {#derivations}

## 1. 每层 receptive field 为什么翻倍

group size 为 2、无重叠时，第一层每个表示汇总 2 个原字符；第二层的每个输入已经汇总 2 个，再合并两个就得到 4 个。经过 $L$ 层：

$$
\operatorname{receptive\_field}=2^L.
$$

一般 group size 为 $g$ 时，理想完整树下为 $g^L$。这里说的是一个输出表示能看到的原字符范围，不是参数量。

## 2. 为什么 block size 8 需要三层

每经过一个 block，时间维从 $T$ 变成 $T/g$。要在 $L$ 层后到 1：

$$
\frac{T}{g^L}=1
\quad\Rightarrow\quad
T=g^L
\quad\Rightarrow\quad
L=\log_g T.
$$

所以 `T=8, g=2` 时 $L=3$。若 `T=6, g=2`，某一层之后无法继续形成完整二叉树，基础实现因此拒绝。

## 3. 参数量与 receptive field 不是同一件事

receptive field 由“每个位置合并多少输入、合并多少层”决定；参数量还受 embedding 宽度、hidden 宽度、每层的投影矩阵影响。

同一个 receptive field 可以配不同 hidden width，得到完全不同的参数量。因此报告层级模型时必须同时写结构、宽度和参数总数。

## 4. 参数共享改变了什么

同一层对 $T/g$ 个位置复用同一个变换。若每个绝对位置各有一个独立变换，参数会随位置数增加，也无法直接把某个位置学到的局部模式用于另一个位置。

层级共享不是“参数更少就一定更好”，而是引入局部复用的归纳偏置；最终效果由公平实验判断。

## 5. 为什么采样代码几乎不用改

无论 flat 还是 hierarchical，模型最终都实现：

$$
(B,T)\rightarrow(B,V).
$$

它仍把固定 context 映射为下一字符分布。自回归循环只依赖这个接口，不需要知道内部是一次展平还是多层合并。

## 自测

1. 对 `(B,8,10)`、hidden width 64，逐层写出 shape。
2. 每个 level 的一个表示分别受到多少原字符影响？
3. 同一层级共享了什么，flat 第一层没有显式共享什么？
4. 本 Lab 没有复制原始 WaveNet 的哪些部分？
5. 为什么 receptive field 扩大不等于参数一定同比增加？
6. 为什么模型内部变化后，自回归采样仍然相同？

## 阅读

- [van den Oord et al., WaveNet (2016)](https://arxiv.org/abs/1609.03499)

完成模型后，重点看 causality、dilation 和 receptive-field growth。论文用于设计背景，不需要先读懂完整音频架构才开始本 Lab。
