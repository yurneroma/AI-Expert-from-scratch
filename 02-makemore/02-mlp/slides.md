---
title: "Context MLP 自学讲义"
subtitle: "从更长上下文到第一个真正的神经网络语言模型"
lang: zh-CN
---

# 这堂课

上一课的 bigram 只看当前字符。本课只增加一个信息来源：**多看前面几个字符**，再用一个小型 MLP 预测下一个字符。

第一次学习不需要先推公式，也不要求先读论文。沿着下面的路线，边理解边写代码：

| 阶段 | 要做什么 | 完成标志 |
|---|---|---|
| [第一阶段：直觉](#intuition) | 看懂滑动 context 和 embedding 的作用 | 能手画 `emma` 的训练样本 |
| [第二阶段：实现](#implementation) | 完成 TODO 1–4 | 数据、参数、forward 和评估 checks 通过 |
| [第三阶段：训练与实验](#training) | 完成 TODO 5–6，比较不同设置 | loss 可训练、能采样，train/dev 结果入表 |
| [第四阶段：公式与解释](#derivations) | 实验后选读 | 能解释参数量、交叉熵和 embedding 梯度 |

# 第一阶段 · 先看懂模型多了什么信息 {#intuition}

## 1. bigram 卡在哪里

bigram 看到当前字符 `a` 时，不知道更前面是 `b` 还是另一个 `a`。只要最后一个字符相同，它就必须使用同一个下一字符分布。

本课把输入改成固定长度的 context。例如 context 长度 `T=3`：

```text
bigram：       a -> 下一个字符
context MLP：e m a -> 下一个字符
```

模型仍然只做下一 token 预测，训练目标和 cross entropy 都没有变。改变的是输入信息和映射函数。

## 2. 一个名字怎样变成训练样本

以 `emma`、`T=3` 为例。开始时没有历史，用边界 `.` 在左侧补齐；每预测一个真实字符，就把窗口向左移动一格：

```text
[., ., .] -> e
[., ., e] -> m
[., e, m] -> m
[e, m, m] -> a
[m, m, a] -> .
```

最后一项仍然学习结束。`X` 的每一行是一个 context，`Y` 保存对应的真实下一个字符：

```text
X.shape = (N, T)
Y.shape = (N,)
```

**先停一下：**用 `ab`、`T=2` 手画三条样本。能画出后再开始 TODO。

## 3. embedding 和 MLP 各做什么

字符编号只是离散地址。embedding 表 `C` 给每个字符一个可学习的向量；相同字符在不同位置使用同一行向量。

把 context 中每个字符的向量取出并拼接，MLP 就可以根据整个窗口预测：

```text
字符编号
  -> 查 embedding
  -> 拼接 context 中的向量
  -> Linear + tanh
  -> Linear
  -> 每个候选字符的 logits
```

embedding 让“在相似上下文里有用的字符”可以共享统计强度；隐藏层则学习 context 中字符的组合。二维 embedding 可以画图观察，但图好看不是评估指标，最终仍要看 dev NLL。

## 4. 信息更多，不等于一定泛化更好

更长 context 能区分更多前缀，但也会扩大第一层参数，并让小数据更容易过拟合。本课的实验不是证明“越大越好”，而是观察：

- train NLL 和 dev NLL 是否一起改善；
- train 很低、dev 很高时，差距有多大；
- 增加 context 和增加隐藏层容量分别带来什么结果。

# 第二阶段 · 从数据写到 forward {#implementation}

## TODO 1 · 先划分完整名字：`split_words`

正确顺序是：

```text
打乱完整名字 -> 划分 train/dev/test -> 每份独立展开 context 样本
```

不能先展开样本再随机划分。否则同一个名字的相邻前缀会同时出现在 train 和 dev，验证集就不再代表“没见过的名字”。

实现时守住四个约定：

- 使用局部、固定 seed 的随机数生成器，结果可复现。
- 不修改传入的 `words`。
- 按比例划分完整名字，三份互不混合。
- 检查比例是否合法。

三个集合的角色不同：

| Split | 用途 |
|---|---|
| train | 更新参数 |
| dev | 选择 context、容量、学习率等设置 |
| test | 所有选择确定后，最后评估一次 |

反复根据 test loss 调参，test 就变成了另一个 dev。

运行：

```bash
uv run pytest -q 02-makemore/02-mlp/check_lab02_mlp.py -k split
```

## TODO 2 · 构造滑动窗口：`build_context_dataset`

对每个名字独立执行：

1. 创建 `block_size` 个边界下标作为初始 context。
2. 把当前 context 加入 `X`，真实字符加入 `Y`。
3. 左移 context，并追加刚才的真实字符。
4. 名字结束后还要预测一次边界。

公开检查使用 `ab`、`T=2`，期望：

```text
X = [[0, 0],
     [0, 1],
     [1, 2]]
Y = [1, 2, 0]
```

两个输出都使用 `torch.long`，shape 分别为 `(N, T)` 和 `(N,)`。

运行：

```bash
uv run pytest -q 02-makemore/02-mlp/check_lab02_mlp.py -k dataset
```

## TODO 3 · 按 shape 建参数：`initialize_parameters`

先写形状，再创建张量：

| 参数 | Shape | 含义 |
|---|---|---|
| `C` | `(V, E)` | 每个字符的 embedding |
| `W1` | `(T*E, H)` | 拼接 context 到隐藏层 |
| `b1` | `(H,)` | 隐藏层偏置 |
| `W2` | `(H, V)` | 隐藏层到字符 logits |
| `b2` | `(V,)` | 输出偏置 |

所有参数都必须是带梯度的叶子张量，并使用同一个局部 generator 保持可复现。

注意这个常见陷阱：

```python
p = torch.randn(shape, requires_grad=True) * scale  # 乘法后不是叶子
```

应先缩放，再调用 `requires_grad_()`。本课先使用脚手架需要的普通随机初始化；为什么输出层通常要更小，留到 Lab 03。

运行：

```bash
uv run pytest -q 02-makemore/02-mlp/check_lab02_mlp.py -k initialize
```

## TODO 4a · 跑一次 forward：`mlp_logits`

设 batch 为 `B`，逐行跟踪 shape：

```text
inputs:  (B, T)
emb:     (B, T, E)       # C[inputs]
embcat:  (B, T*E)        # 只拼接 token 和 embedding 维
h:       (B, H)          # tanh(embcat @ W1 + b1)
logits:  (B, V)          # h @ W2 + b2
```

`reshape(B, T*E)` 必须保留 batch 维。即使 `B=1`，输出也应该是 `(1, V)`，不能无意 squeeze 成 `(V,)`。

`C[inputs]` 是可微查表：backward 会把梯度送回被选中的 embedding 行；同一行被重复选中时，贡献会累加。

运行：

```bash
uv run pytest -q 02-makemore/02-mlp/check_lab02_mlp.py -k logits
```

## TODO 4b · 评估时不保留训练图：`evaluate_mlp_loss`

训练仍用真实目标计算平均 cross entropy：

```python
torch.nn.functional.cross_entropy(logits, targets)
```

评估时在 `torch.no_grad()` 下 forward，并返回 Python `float`。这样既节省内存，也明确评估不会更新参数。

运行：

```bash
uv run pytest -q 02-makemore/02-mlp/check_lab02_mlp.py -k evaluate
```

### 第二阶段完成检查

在不看代码的情况下，对 `B=32, T=3, E=10, H=200, V=27` 写出从 `X` 到 `logits` 的每个 shape，并指出哪个参数随 context 长度增长。

# 第三阶段 · 训练、采样和比较 {#training}

## TODO 5 · mini-batch 训练：`train_mlp`

每一步只走这条闭环：

```text
用固定 seed 的 generator 抽 batch 下标
-> 取 inputs 和 targets
-> 清空所有参数的旧梯度
-> forward + cross entropy
-> backward
-> 在 no_grad 中更新所有参数
-> 记录标量 loss
```

mini-batch 梯度更便宜，但有噪声，所以 batch loss 曲线不要求单调下降。判断训练时结合完整 train/dev loss，不要因为某一个 batch 上升就认定发散。

训练完成后，至少比较训练前后的完整数据集 loss。

运行：

```bash
uv run pytest -q 02-makemore/02-mlp/check_lab02_mlp.py -k train
```

## TODO 6 · 用同一个 context 约定采样：`sample_mlp`

生成从 `[0] * block_size` 开始：

```text
把当前 context 放进 batch 维，形状变成 (1, T)
-> no_grad 下计算 logits
-> softmax 后抽一个字符
-> 抽到边界就结束
-> 否则记录字符，左移 context 并追加新字符
-> 达到 max_length 也停止
```

训练和采样必须使用相同的 `stoi`、`block_size` 和左填充约定。不要把 logits 直接当成概率传给抽样函数。

运行：

```bash
uv run pytest -q 02-makemore/02-mlp/check_lab02_mlp.py -k sample
uv run pytest -q 02-makemore/02-mlp/check_lab02_mlp.py
```

## 必做实验：先预测，再运行

使用同一份 word split，并把配置、seed、训练步数记录到 `notes.md`。

### A. 改变 context 长度

固定其他设置，比较：

```text
block_size = 1, 2, 3, 5
```

`block_size=1` 在概念上退化为只看一个字符，但参数共享方式与 bigram 权重表不同。观察 train/dev NLL，而不是只挑生成得最好的样本。

### B. 改变容量

比较：

- small：`embedding_dim=2, hidden_dim=50`
- larger：`embedding_dim=10, hidden_dim=200`

记录参数量、train/dev NLL 和 loss 曲线：

| 现象 | 优先假设 |
|---|---|
| train 高、dev 高 | 欠拟合或优化失败 |
| train 低、dev 明显高 | 过拟合 |
| 两者一起改善 | 有效容量或更好的优化 |

这些是指导下一步实验的假设，不是只凭一条曲线就能证明的结论。

## 常见失败

| 现象 | 先查什么 |
|---|---|
| dev 异常好或差得不可信 | 是否先展开样本再随机 split |
| reshape 报错或 B=1 才出错 | batch 维是否被压掉 |
| 改 block size 后第一层报错 | `W1` 输入宽度是否仍是 `T*E` |
| 参数没有梯度 | 是否创建了非叶子张量，或评估图被 detach 后复用 |
| 每次采样都一样 | 是否 argmax，或每个名字重置同一 seed |
| test 越调越好 | 是否在用 test 选择超参数 |

# 第四阶段 · 公式与解释（实验后选读） {#derivations}

## 1. 条件概率只改变了条件

bigram 使用：

$$
P(x_t\mid x_{t-1}).
$$

固定 context MLP 使用：

$$
P(x_t\mid x_{t-T},\ldots,x_{t-1}).
$$

它仍然是一步条件分布；整名概率仍由各步相乘得到。模型没有换成别种训练目标，只是看见了更多历史。

## 2. 参数量从哪里来

embedding、两层 affine 和偏置相加：

$$
\underbrace{VE}_{C}
+\underbrace{(TE)H}_{W1}
+\underbrace{H}_{b1}
+\underbrace{HV}_{W2}
+\underbrace{V}_{b2}.
$$

当 `T` 增加时，`W1` 随之增大；当 `E` 或 `H` 增大时，多项参数同时变化。比较结构时必须报告参数量，不能把“大模型 dev 更好”直接归因于 context 设计。

## 3. cross entropy 仍然只看真实目标

对 batch 中第 $m$ 个样本，真实类别为 $y_m$，预测分布为 $p^{(m)}$：

$$
L=-\frac1B\sum_{m=1}^{B}\log p^{(m)}_{y_m}.
$$

这与 bigram 的 NLL 相同。MLP 改变的是 logits 的计算方式，没有改变“真实下一个字符拿到的概率越高，loss 越低”。

## 4. embedding lookup 的梯度为什么会累加

前向中，$C$ 的第 $i$ 行可能被多个位置同时选中。每个选中位置都是同一份参数的一条不同使用路径，所以反向时把各路径梯度相加，而不是互相覆盖。

这也是后面手动 backward 中 `index_add_` 存在的原因。当前只需理解：**查表不是把参数复制成互不相关的新参数，所有使用仍指回同一行。**

## 自测

1. 为什么 word split 必须先于 context 样本展开？
2. `C` 的一行代表什么？为什么同一行会在不同 context 位置复用？
3. 哪个参数直接随 context 长度增长？
4. train loss 改善而 dev loss 变差说明什么？还不能说明什么？
5. 为什么训练和采样必须共用完全相同的 context 约定？
6. `block_size=1` 在输入信息上回到了什么模型？

## 阅读

- [Bengio et al., A Neural Probabilistic Language Model (2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

先看模型结构图和 distributed representation 的动机即可。论文用词作为 token，本 Lab 用字符；不需要读完整篇才开始写代码。
