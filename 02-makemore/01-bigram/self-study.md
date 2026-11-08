---
title: "Bigram 自学讲义"
subtitle: "先直觉，再代码与训练，最后看推导"
lang: zh-CN
---

# 这堂课

**第一次读，不需要先懂公式。** 你要做的只是：读一份名字列表，学会“当前字符后面通常接什么”，再生成几个拼写习惯相似的新名字。

本 Lab 会写两个能力相同、训练方式不同的模型：

- **计数 bigram**：数出每种相邻字符出现几次，再把次数变成概率。它的“训练”就是计数和归一化。
- **神经 bigram**：把同样一张表放进权重矩阵，用交叉熵和梯度下降训练。它是在为后面的 MLP、Transformer 练训练闭环。

公式不会消失，但放到了最后。先让程序跑起来，再回头看为什么这些规则是对的。

## 学习路线

| 阶段 | 你要做什么 | 完成标志 |
|---|---|---|
| [第一阶段：直觉](#intuition) | 只读，不写公式 | 能用自己的话说清 bigram 的输入、输出和局限 |
| [第二阶段：计数模型](#count-model) | 边读边做 TODO 1–5 | 能建表、评分、采样，对应 checks 通过 |
| [第三阶段：神经模型与训练](#neural-model) | 做 TODO 6–7，跑完整实验 | loss 下降，能生成名字，全部 checks 通过 |
| [第四阶段：公式推导](#derivations) | 实现完成后选读 | 能解释规则为什么成立，而不是只会照抄 |

每一阶段都可以单独停下来。不要把“读完所有推导”当作开始写代码的门票。

---

# 第一阶段 · 先用直觉看懂 bigram {#intuition}

## 1. 它到底在学什么

假设训练数据只有：

```python
words = ["ab", "aa"]
```

我们不希望程序只会原样抄出 `ab` 和 `aa`，也不希望它从字母表里乱抽。希望它学到一些局部规律，再重新组合。

bigram 把任务缩成一个反复回答的问题：

> **现在看到的字符是什么？下一个字符可能是什么？**

例如当前字符是 `a`，模型可能认为：

```text
下一个是 . ：约 1/3
下一个是 a ：约 1/3
下一个是 b ：约 1/3
```

模型输出的不是唯一答案，而是一组概率。这样它既能按概率生成不同名字，也能给真实名字打分。

这里的 token 就是字符。bigram 的 bi 表示一次看相邻的两个 token：左边是输入，右边是预测目标。

## 2. 为什么要有 `.`

模型还需要知道两件事：名字怎样开始，以及何时结束。本 Lab 用 `.` 同时表示这两个边界。

```text
名字 ab： . a b .
训练边：  (. → a), (a → b), (b → .)

名字 aa： . a a .
训练边：  (. → a), (a → a), (a → .)
```

- `. → a` 让模型学习首字符。
- `b → .` 和 `a → .` 让模型学习结束。
- 长度为 2 的名字会得到 3 条训练边；一般地，长度为 `L` 的名字得到 `L + 1` 条。

每个名字要单独加边界。不能把 `ab` 和 `aa` 先拼成 `abaa`，否则会凭空制造一条 `b → a`，还会丢失真实的结束和开始。

## 3. “训练”可以先只是数数

把六条边按“当前字符”分组：

| 当前字符 | 看到的下一个字符 |
|---|---|
| `.` | `a`、`a` |
| `a` | `b`、`a`、`.` |
| `b` | `.` |

所以看到 `a` 时，三种后继各出现一次，各给 `1/3` 就很自然。看到 `b` 时，唯一见过的后继是 `.`，于是直接预测结束。

这就是计数 bigram 的核心：**数转移次数，再把每一行变成概率。**

## 4. 它怎样生成训练集中没有的名字

从 `.` 开始，根据当前这一行的概率抽下一个字符；抽到普通字符就把它作为新的当前字符，抽到 `.` 就停止。

一条可能的路径是：

```text
. → a → a → b → .
```

它生成了 `aab`。训练列表里没有这个名字，但每一步都使用了训练列表里出现过的局部规律。

训练与生成不要混在一起：

- **训练**：看数据里真实的下一条边。
- **生成**：看模型自己刚抽出来的字符。

## 5. 它天生学不会什么

bigram 只看当前一个字符。`ba` 和 `aa` 的最后一个字符都是 `a`，所以它无法因为更前面的字符不同，就给出不同的下一字符分布。

这不是训练不够久，而是输入信息不够。下一课的 context MLP 才会解决这个问题。

**第一阶段检查：** 不看上文，能否回答下面三句？

1. 输入是当前字符，输出是什么？
2. 为什么 `ab` 有三条训练边，而不是两条？
3. 为什么 bigram 能拼出新名字，却不能利用更长上下文？

可以回答后，直接开始写代码。后面的内容按 TODO 分段阅读即可。

---

# 第二阶段 · 写计数模型 {#count-model}

本阶段完成 `lab01_bigram.py` 的 TODO 1–5。不要一次读完整节再动手；每完成一个函数，就跑对应检查。

## TODO 1 · 把字符变成表的位置：`build_vocab`

矩阵不认识字符，只认识行列下标。因此需要两个字典：

```text
stoi = {".": 0, "a": 1, "b": 2}
itos = {0: ".", 1: "a", 2: "b"}
```

编号只是地址，不表示字符的大小或距离。本 Lab 约定：

- 边界 `.` 固定为 0。
- 其余字符排序后编号，保证每次结果一致。
- `stoi` 和 `itos` 必须互为逆映射。

在仓库根目录运行：

```bash
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k vocab
```

通过后，在 `notes.md` 记下词表大小 `V`。这个小例子里 `V = 3`，完整名字数据通常是 26 个字母加边界，即 `V = 27`。

## TODO 2 · 把名字数成一张表：`build_bigram_counts`

创建一张 `V × V` 的整数表。约定不变：

- **行**：当前字符。
- **列**：下一个字符。
- **格子**：这条边出现了多少次。

对 `ab`、`aa`，结果必须是：

| 当前 \ 下一个 | `.`（0） | `a`（1） | `b`（2） |
|---|---:|---:|---:|
| `.`（0） | 0 | 2 | 0 |
| `a`（1） | 1 | 1 | 1 |
| `b`（2） | 1 | 0 | 0 |

例如 `C[0, 1] = 2`，因为两个名字都产生 `. → a`。`a` 这一行的总数是 3，因为它在两个名字中一共作为当前字符出现了三次。

运行：

```bash
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k counts
```

如果表对不上，先打印单个 `ab` 的三条边，不要直接在完整数据里找错。

## TODO 3 · 把次数变成概率：`counts_to_probabilities`

每一行要独立变成概率：格子除以**这一行的总数**。本阶段先记住这个规则，它的最大似然证明在[后面](#mle)。

用代码表达时，关键是保留行维度：

```python
row_totals = counts.sum(dim=1, keepdim=True)
probabilities = counts / row_totals
```

`keepdim=True` 让分母保持 `(V, 1)`，于是每一列都使用自己所在行的分母。方阵里写错广播方向也可能不报错，所以必须检查：

```python
probabilities.sum(dim=1)
```

每一行都应该为 1；整张表的和不是 1，而是有定义的行数之和。

### 平滑：给没见过的转移留一点概率

未平滑时，`b → a` 的次数是 0，模型就会认为它绝对不可能。新名字只要包含一条这样的边，概率就会变成 0。

加法平滑先给每个格子补上同一个 `smoothing`，再做行归一化：

```text
用于归一化的格子 = 原计数 + smoothing
```

若一行有 `V` 个格子，分母自然增加 `V × smoothing`。只给分子加数、不调整分母，行和就不再是 1。

本例 `smoothing=1` 时：

| 当前 \ 下一个 | `.` | `a` | `b` |
|---|---:|---:|---:|
| `.` | 1/5 | 3/5 | 1/5 |
| `a` | 1/3 | 1/3 | 1/3 |
| `b` | 1/2 | 1/4 | 1/4 |

平滑还有另一个作用：完全没见过的行原本会得到 `0/0`，加平滑后会变成均匀分布。`smoothing=0` 时空行没有唯一答案，需要自己明确处理，不要把 NaN 当作一个分数。

运行：

```bash
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k probabilities
```

## TODO 4 · 给模型一个分数：`average_nll`

训练时，每条样本都有真实的下一个字符。我们只看模型给这个真实字符分了多少概率：

| 真实字符拿到的概率 `p` | 单次损失 `-log(p)` | 直觉 |
|---:|---:|---|
| 1 | 0 | 完全猜对，而且非常确定 |
| 0.5 | 0.693 | 只分了一半信心 |
| 0.1 | 2.303 | 很不看好真实结果 |
| 0 | 无穷 | 认为真实结果不可能 |

NLL 就是“负对数似然”。当前实现只需记住：

1. 对每条真实边取 `P[current, next]`。
2. 取 `-log`。
3. 对所有边求平均。

本例共有 6 条边，所以分母是 6，不是名字数 2，也不是正文字符数 4。开始和结束边都参与评分。

两个可用来定位 bug 的数值：

- 完全均匀的 3 类预测，平均 NLL 是 `log(3) ≈ 1.098612`。
- 本例未平滑计数表在训练数据上的平均 NLL 是 `0.549306`，越低越好。

先不推导这些数为什么是最优；现在只要能用它们检查实现。

运行：

```bash
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k nll
```

## TODO 5 · 从表里抽新名字：`sample_bigram`

生成循环只需要下面这张伪代码：

```text
每个新名字都从当前字符 . 开始
重复：
    取概率表的当前行
    用 torch.multinomial 按这一行抽一个下标
    如果抽到 .：结束
    否则：记录字符，并把它变成新的当前字符
达到 max_length 也要停止
```

注意三个区别：

- 按概率抽样，不是每次取 argmax；否则固定模型容易永远走同一条路。
- 一个生成器连续生成多个名字，不要每生成一个名字就重置成同一个种子。
- 达到 `max_length` 是程序截断，不是模型预测了结束。

运行：

```bash
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k sample
```

### 第二阶段完成检查

到这里，第一个模型已经“训练”完了：计数就是估计参数，归一化就是得到概率表。你应当能：

- 从 `ab`、`aa` 手画计数表。
- 解释为什么每一行分别归一化。
- 用真实边算 NLL，用模型自己抽出的边生成名字。
- 说出平滑解决的是零概率，而不是让训练集 NLL 一定更低。

如果还有一处说不清，再回看对应 TODO 的小例子即可，不需要提前读公式附录。

---

# 第三阶段 · 写神经版并训练 {#neural-model}

## 1. 计数模型已经能工作，为什么还要神经版

计数版可以直接算出答案，神经版并不会因为用了梯度下降就突然看得更远。两者仍然只看当前字符，目标仍是同一张条件概率表。

神经版的价值是练习一套可复用的接口：

```text
输入编号
  → 找到这一输入对应的参数行
  → 输出每个候选字符的 logits
  → cross entropy 得到 loss
  → backward 得到梯度
  → 更新权重
```

以后把权重表换成 embedding、隐藏层或 Transformer，这个训练闭环仍然适用，但不能再靠“计数除以行总数”一步求出答案。

## TODO 6 · 把训练边铺成 batch：`build_bigram_dataset`

计数阶段可以逐名字处理；训练网络时，把所有边收进两个一维张量：

```text
inputs  = [0, 1, 2, 0, 1, 1]
targets = [1, 2, 0, 1, 1, 0]
```

第 `m` 对 `(inputs[m], targets[m])` 就是第 `m` 条真实边。两者都使用整数类别编号，shape 都是 `(M,)`；本例 `M = 6`。

运行：

```bash
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k dataset
```

## TODO 7a · 从权重表得到 loss：`neural_bigram_loss`

权重 `weights` 仍是 `V × V`：

- 行仍是当前字符。
- 列仍是候选的下一个字符。
- 格子现在叫 **logit**，是可以为正、为负的分数，不是概率。

对一批输入：

```python
logits = weights[inputs]
```

`inputs` 的 shape 是 `(B,)`，得到的 `logits` 是 `(B, V)`。每个样本选中自己的一行；one-hot 乘法也只是另一种“选行”写法。

接下来不用先手动写 softmax。`torch.nn.functional.cross_entropy(logits, targets)` 会在内部完成 softmax 和真实类别的 NLL，并使用更稳定的数值实现。

```text
weights: (V, V)
inputs:  (B,)
logits:  (B, V)
targets: (B,)
loss:    标量
```

注意：

- 传原始 logits，不要先对它们做 softmax 再传入。
- `targets` 是整数类别编号，不需要手工 one-hot。
- 可选正则项加在数据 loss 后面；含正则的 objective 和纯 NLL 不是同一个数。

零初始化时，每一行的 logits 都相同，所以预测是均匀的。本例初始数据 loss 应接近 `log(3) ≈ 1.098612`。

运行：

```bash
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k neural_loss
```

## TODO 7b · 真正训练：`train_neural_bigram`

每一轮只做五件事：

```text
清空旧梯度
根据当前 weights 算 logits 和 loss
loss.backward()
按 learning_rate 更新 weights
记录本轮 loss
```

对应职责：

- `weights` 是要求梯度的叶子参数。
- 默认梯度会累加，所以下一轮前要 `weights.grad = None` 或清零。
- 参数更新放在 `torch.no_grad()` 中，避免更新本身进入计算图。
- `loss.item()` 只用于记录；它不能代替原 loss 做 `backward()`。
- 返回 detached 的权重和长度为 `steps` 的 loss history。

这个模型没有隐藏层，零初始化可以学习，不需要套用“多层网络必须随机打破对称”的结论。

运行：

```bash
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k neural_training
```

首先确认 `history[-1] < history[0]`。如果 loss 不降，按顺序查：目标是否对齐、旧梯度是否清空、是否真的更新了同一份权重、学习率是否合适。

## 2. 跑完整实验

全部函数完成后：

```bash
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py
uv run python 02-makemore/01-bigram/run_lab01.py
```

实验会比较不同平滑强度的计数模型和神经模型，输出生成样本，并保存到 `artifacts/metrics.json`。默认优先使用完整 `data/names.txt`，不存在时使用小数据；抄结果前先核对输出里的数据路径、单词数和词表大小。

把结果填入 `notes.md`，至少观察：

1. 均匀 baseline 是否接近 `log(V)`。
2. 平滑变强时，训练 NLL 和生成样本怎样变化。
3. 神经版与计数版是否学到相近的分布；有限步数和正则会让它们不完全相同。
4. 神经版输出的 `final_objective` 可能含正则，不能直接当作纯 NLL 与计数版比较。

当前脚本是在同一份 `words` 上拟合并评分，所以报告的是**训练 NLL**，不是泛化能力。真正的验证集应先按完整名字划分，再分别拆转移，不能先随机拆边。

### 第三阶段完成检查

现在你已经完成了本 Lab 的代码和训练。先用自己的话解释：

- 为什么 `weights[inputs]` 等价于 one-hot 选行？
- 为什么 cross entropy 在这里就是“真实下一个字符的 NLL”？
- 训练时使用真实目标，生成时为什么改用模型刚抽出的字符？
- 为什么训练更久也不能让 bigram 利用更长的上下文？

**如果目标是先会做，你可以在这里停止阅读，去完成 Lab。** 下一阶段的推导用于回答“为什么”，不再是写代码的前置条件。

## 3. 出问题时，先回到小例子

| 现象 | 先查什么 |
|---|---|
| 行和不是 1 | `dim=1`、`keepdim=True`、分母是否确实是行总数 |
| 概率表有 NaN | 无平滑时是否出现空行 `0/0` |
| 新名字 NLL 无穷 | 某条真实边是否被未平滑表赋了零概率 |
| 每次生成都一样 | 是否用了 argmax，或每个名字都重置了随机种子 |
| loss 有数但没有梯度 | 是否 detach、是否对 `.item()` 的数求导、是否换了新张量 |
| loss 不降或乱跳 | 输入目标是否对齐、梯度是否累加、学习率是否过大 |
| 神经版与计数版不完全相同 | 训练步数、正则、零概率的极限，以及比较的是否都是纯 NLL |

调试顺序固定为：**一个名字 → 六条边 → 一张小表 → 一个小 batch → 完整语料。** 不要只看最后一个总数。

---

# 第四阶段 · 公式推导（实现后选读） {#derivations}

这部分保留完整推导。可以按问题选读，不必一次读完。

| 想弄清楚的问题 | 看哪一节 |
|---|---|
| 为什么逐步概率相乘就是整名概率？ | [链式法则与 bigram 近似](#chain-rule) |
| 为什么次数除以行总数就是最大似然？ | [最大似然](#mle) |
| 为什么 loss 要取 log、加负号、再平均？ | [NLL](#nll-derivation) |
| 平滑为什么等价于向均匀分布靠拢？ | [加法平滑](#smoothing-derivation) |
| logits 怎样变成概率，为什么要稳定实现？ | [softmax 与交叉熵](#softmax-derivation) |
| 一次梯度更新到底改了什么？ | [梯度推导](#gradient-derivation) |
| 两种 bigram 为什么指向同一个分布？ | [模型等价关系](#equivalence) |


## 4.1 整名概率：链式法则与 bigram 近似 {#chain-rule}

条件概率

$$
P(B \mid A) = \frac{P(A, B)}{P(A)}
$$

把分母乘回去，就得到 $P(A, B) = P(A)P(B \mid A)$。连续使用这个定义：

$$
P(A, B, C) = P(A)P(B \mid A)P(C \mid A, B).
$$

这就是概率链式法则，与反向传播里的求导链式法则不是同一件事。

暂时用 `BOS` 和 `EOS` 区分开始与结束，名字 `ab` 的完整事件可以拆成：

$$
\begin{aligned}
P(a, b, \mathrm{EOS} \mid \mathrm{BOS})
&= P(a \mid \mathrm{BOS}) \\
&\quad \cdot P(b \mid \mathrm{BOS}, a) \\
&\quad \cdot P(\mathrm{EOS} \mid \mathrm{BOS}, a, b).
\end{aligned}
$$

起点是给定条件，不必再乘“起点出现的概率”；结束是模型要预测的事件，必须乘进去。

完整前缀往往只出现一两次，难以为每一种前缀估计一个分布。bigram 因此做了一个建模选择：

$$
P(c_t \mid c_1, \ldots, c_{t-1}) \approx P(c_t \mid c_{t-1}).
$$

这叫一阶 Markov 假设。近似号很重要：缩短历史不是链式法则推出来的定理，而是为了减少稀疏性而接受的信息损失。

在本 Lab 的模型里，$c_0 = \texttt{.}$，$c_{L+1} = \texttt{.}$，名字概率定义为：

$$
P_\theta(c_1 \ldots c_L \text{ 并结束})
= \prod_{t=1}^{L+1} P_\theta(c_t \mid c_{t-1}).
$$

所以 `ab` 对应：

$$
P_\theta(\texttt{ab})
= P_\theta(a \mid \texttt{.})P_\theta(b \mid a)P_\theta(\texttt{.} \mid b).
$$

## 4.2 为什么计数除以行总数：最大似然 {#mle}

### 先看 `a` 这一行

当前字符固定为 `a` 时，三个观测的后继分别是 `b`、`a`、`.`。设这一行的概率为 $(p_., p_a, p_b)$，数据固定，选择不同的概率就会得到不同的似然：

$$
\mathcal L_a = p_b p_a p_. .
$$

比较两组候选：

$$
(1/3)^3 \approx 0.037,
\qquad
0.8 \times 0.1 \times 0.1 = 0.008.
$$

第二组偏爱一个字符，却让另外两个真实观测更难发生。最大似然要找的是：在所有合法分布里，让固定观测最容易发生的那一组。

令 $p_a = u$、$p_b = v$、$p_. = 1-u-v$，先假设三者都为正。对数单调递增，所以最大化下式不改变最优解的位置：

$$
f(u, v) = \log u + \log v + \log(1-u-v).
$$

令两个偏导为零：

$$
\frac{1}{u} = \frac{1}{1-u-v},
\qquad
\frac{1}{v} = \frac{1}{1-u-v}.
$$

于是 $u = v = 1-u-v$。三种概率相等且和为 1，所以各为 $1/3$。目标在正数区域是凹的；任何真实观测的概率碰到 0，似然就变成 0，因此这里是最大值。

开始行有两个 `. → a`，似然为 $p_a^2$，最大值在 $p_a=1$。`b` 行只有一个 `b → .`，最大值在 $p_.=1$。于是未平滑表为：

$$
P_{\mathrm{MLE}} =
\begin{pmatrix}
0 & 1 & 0 \\
1/3 & 1/3 & 1/3 \\
1 & 0 & 0
\end{pmatrix}.
$$

### 任意计数的一般证明

固定某一行 $i$，把概率写成 $p_j$，计数写成 $c_j = C_{ij}$，行总数为 $n = \sum_j c_j > 0$。相同后继的因子可以合并：

$$
\mathcal L_i = \prod_j p_j^{c_j},
\qquad
\log \mathcal L_i = \sum_j c_j \log p_j.
$$

先假设所有 $c_j > 0$。需要最大化这个目标，同时满足 $p_j \ge 0$ 和 $\sum_j p_j = 1$。用辅助变量 $\mu$ 把概率和约束加入目标：

$$
\Phi(p, \mu) = \sum_j c_j \log p_j + \mu\left(1-\sum_j p_j\right).
$$

对每个 $p_j$ 求导并令其为零：

$$
\frac{c_j}{p_j} - \mu = 0
\quad\Rightarrow\quad
p_j = \frac{c_j}{\mu}.
$$

对 $\mu$ 求导会拿回约束。把所有类别加起来：

$$
1 = \sum_j p_j = \frac{n}{\mu}
\quad\Rightarrow\quad
\mu = n
\quad\Rightarrow\quad
\boxed{p_j = \frac{c_j}{n}}.
$$

这就是“次数除以行总数”，不是单纯为了让行和变成 1 的技巧。

若某个 $c_j=0$，未平滑最大似然会把该类别概率设为 0；对正计数类别重复内部推导即可，不能在 $p_j=0$ 处直接套 $c_j/p_j$。整行 $n=0$ 时没有观测依赖这行，任何合法分布都同样符合数据，最大似然给不出唯一答案。

整份数据的 log-likelihood 是：

$$
\log \mathcal L = \sum_i\sum_j C_{ij}\log P_{ij}.
$$

不同行不共享参数，各有自己的“和为 1”约束，所以可以逐行求最优。后面的网络一旦通过共享权重连接不同上下文，就不能再逐行独立求解。

## 4.3 为什么 NLL 要取对数、加负号、再平均 {#nll-derivation}

用未平滑表：

$$
P(\texttt{ab}) = 1 \times \frac13 \times 1 = \frac13,
\qquad
P(\texttt{aa}) = 1 \times \frac13 \times \frac13 = \frac19.
$$

把两个名字看成两次独立生成，数据似然为：

$$
\mathcal L = P(\texttt{ab})P(\texttt{aa}) = \frac1{27}.
$$

这里相乘的是不同名字；一个名字内部使用条件概率相乘，不要求相邻字符独立。

很多小于 1 的概率连乘会很快下溢。例如 $0.1^{1000}=10^{-1000}$。自然对数把乘积变成和：

$$
\log \mathcal L = \sum_{m=1}^{M}\log P_{x_m,y_m}.
$$

对数严格递增，最大化似然和最大化 log-likelihood 是同一个优化问题。训练框架通常最小化 loss，所以加负号；再除以预测次数 $M$，得到每条转移的平均损失：

$$
\boxed{
L_{\mathrm{NLL}}
= -\frac1M\sum_{m=1}^{M}\log P_{x_m,y_m}.
}
$$

本例 $M=6$：

$$
L_{\mathrm{NLL}}
= -\frac16\log\frac1{27}
= \frac{\log 27}{6}
= \frac{\log3}{2}
\approx 0.549306.
$$

若 $V$ 个候选总是均匀分配概率，每次损失都是 $-\log(1/V)=\log V$，平均后仍是 $\log V$。它是“完全均匀预测”的固定对照，不是理论最小值；随机模型若自信地猜错，loss 可以更高。

可选的困惑度只是换一种读法：

$$
\mathrm{PPL} = e^{L_{\mathrm{NLL}}}.
$$

均匀 $V$ 类预测的 PPL 是 $V$。比较不同实验时，数据、token 切法、边界约定和对数底必须一致。

## 4.4 加法平滑为什么有效 {#smoothing-derivation}

未平滑表把没见过的转移设为零。新名字 `ba` 包含没见过的 `b → a`，因此：

$$
P(\texttt{ba})
= P(b\mid\texttt{.})P(a\mid b)P(\texttt{.}\mid a)
= 0.
$$

给每个格子加同一个 $\alpha>0$，一行有 $V$ 格，分母就要增加 $V\alpha$：

$$
\boxed{
P^{(\alpha)}_{ij}
= \frac{C_{ij}+\alpha}{n_i+V\alpha}.
}
$$

$\alpha=1$ 是 add-one/Laplace 平滑，$\alpha$ 也可以是小数。本例加一之后：

$$
P^{(1)} =
\begin{pmatrix}
1/5 & 3/5 & 1/5 \\
1/3 & 1/3 & 1/3 \\
1/2 & 1/4 & 1/4
\end{pmatrix}.
$$

对有观测的行，可以拆成：

$$
\frac{C_{ij}+\alpha}{n_i+V\alpha}
= \lambda_i\frac{C_{ij}}{n_i} + (1-\lambda_i)\frac1V,
\qquad
\lambda_i = \frac{n_i}{n_i+V\alpha}.
$$

所以平滑是在经验分布和均匀分布之间插值：样本越多，经验分布权重越大；$\alpha$ 越大，越靠近均匀。空行 $n_i=0$ 不能使用这个含 $C_{ij}/n_i$ 的拆分，但原公式直接给出 $1/V$。

未平滑最大似然已经把训练 NLL 降到最低，因此平滑通常会让训练分数稍差，换来新转移不再是零概率。本例：

| $\alpha$ | 训练集 `ab, aa` 平均 NLL | 留出示例 `ba` 平均 NLL |
|---:|---:|---:|
| 0 | 0.549306 | $+\infty$ |
| 0.1 | 0.607472 | 2.266352 |
| 1 | 0.835106 | 1.364782 |
| 10 | 1.048931 | 1.131055 |

这只是极小例子的核对，不足以给真实任务选 $\alpha$；真实选择要看验证集。

## 4.5 logits、softmax 与交叉熵 {#softmax-derivation}

字符编号只是类别地址。one-hot 用开关表示类别：

$$
e_.=(1,0,0),\quad e_a=(0,1,0),\quad e_b=(0,0,1).
$$

若 $W$ 的行是当前字符、列是下一个字符，则 $e_aW$ 恰好选中 `a` 那一行。一个 batch 中：

$$
X\in\mathbb R^{B\times V},\quad
W\in\mathbb R^{V\times V},\quad
Z=XW\in\mathbb R^{B\times V}.
$$

实现里的 `W[inputs]` 与这个乘法等价，也会把同一行被多次选中的梯度累加起来。

权重更新不会自动保证“非负且和为 1”，所以用 softmax 把任意分数 $z_j$ 变成概率：

$$
\boxed{
p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}.
}
$$

例如 $z=(0,1,2)$：

$$
e^z\approx(1,2.718,7.389),
\qquad
p\approx(0.090,0.245,0.665).
$$

真正起作用的是分数差：

$$
\frac{p_j}{p_k}=e^{z_j-z_k}.
$$

整行同时加常数 $c$，分子分母都多一个 $e^c$，概率不变。因此 `(0,1,2)` 与 `(100,101,102)` 表示同一分布，权重数值并不唯一。

直接对很大的 logits 求指数可能溢出。先减去行最大值 $m=\max_k z_k$：

$$
p_j = \frac{e^{z_j-m}}{\sum_k e^{z_k-m}}.
$$

训练时还可以留在对数空间：

$$
\log p_j = z_j-\log\sum_k e^{z_k},
\qquad
\log\sum_k e^{z_k}
= m+\log\sum_k e^{z_k-m}.
$$

这就是 `log_softmax` 和 `cross_entropy` 采用稳定实现的原因。

真实类别为 $y$ 时，单个样本的 NLL 是：

$$
\begin{aligned}
\ell(z,y)
&=-\log p_y\\
&=-z_y+\log\sum_k e^{z_k}.
\end{aligned}
$$

若把一次真实观测写成 one-hot 目标 $q$，交叉熵为：

$$
H(q,p)=-\sum_j q_j\log p_j=-\log p_y.
$$

所以在“一个正确类别、不加权、不做 label smoothing”的本 Lab 中，每个样本的交叉熵就是它的 NLL，batch 平均就是平均 NLL。

对 `z=(0,1,2)`、真实类别 `a`（编号 1）：

$$
\ell=-1+\log(1+e+e^2)\approx1.407606.
$$

虽然 `b` 的 logit 最大，loss 取的仍是真实字符 `a` 的概率。

## 4.6 梯度为什么是预测减目标 {#gradient-derivation}

从单个样本的损失出发：

$$
\ell=-z_y+\log\sum_k e^{z_k}.
$$

对任意 $z_j$ 求导。第一项只在 $j=y$ 时贡献 $-1$，第二项的导数是 softmax 概率，因此：

$$
\boxed{
\frac{\partial\ell}{\partial z_j}
= p_j-\mathbf 1[j=y].
}
$$

向量形式就是 $p-q$。对 `z=(0,1,2)`、真实类别 `a`：

$$
\nabla_z\ell
\approx(0.090,-0.755,0.665).
$$

梯度下降执行 $z\leftarrow z-\eta\nabla_z\ell$：真实类别的分数上升，其他类别的分数下降。梯度和为 0，也对应“整行一起平移不改变概率”。

对输入编号为 $i$ 的样本，$z_j=W_{ij}$。batch 平均后：

$$
\frac{\partial L}{\partial W_{ij}}
= \frac1B\sum_{m:x_m=i}
\left(p_j^{(m)}-\mathbf1[y_m=j]\right).
$$

所以数据 loss 只更新 batch 里被选中的行；没被选择的行没有数据梯度，但仍可能收到正则梯度。

### 六个样本、零初始化的一次更新

$W=0$ 时，每行预测都是 $(1/3,1/3,1/3)$。同一行上的样本合并后：

$$
\frac{\partial L}{\partial W_{ij}}
=\frac{n_iP_{ij}-C_{ij}}{M}.
$$

代入本例 $M=6$：

$$
\nabla_WL=
\begin{pmatrix}
1/9 & -2/9 & 1/9\\
0 & 0 & 0\\
-1/9 & 1/18 & 1/18
\end{pmatrix}.
$$

`a` 行梯度全为 0，因为均匀预测已经等于这一行的经验分布。开始行把概率挪向 `a`，`b` 行把概率挪向结束。学习率为 1 时，一次更新让平均 NLL 从约 `1.098612` 降到 `1.010797`；文末代码会用自动微分核对这个结果。

## 4.7 两种模型何时等价，何时不同 {#equivalence}

计数版直接保存 $P(j\mid i)$，神经版保存 $W_{ij}$，再对每行做 softmax。若目标表每一项都严格大于 0，取：

$$
W_{ij}=\log P_{ij},
$$

就有：

$$
\operatorname{softmax}(W_i)_j
= \frac{P_{ij}}{\sum_kP_{ik}}
= P_{ij}.
$$

所以神经 bigram 能精确表示任何严格为正的条件概率表。每行加同一个常数，表示的分布仍不变。

若某格目标概率为 0，有限 logits 经 softmax 不能得到精确的 0，只能让相对分数趋向负无穷。因此神经模型可以不断接近未平滑最大似然，但不一定在有限权重处正好到达边界。

对 $n_i>0$ 的行，令数据梯度为零：

$$
n_iP_{ij}=C_{ij}
\quad\Rightarrow\quad
P_{ij}=\frac{C_{ij}}{n_i}.
$$

这说明：不加正则时，两种模型对见过的输入行指向同一个经验条件分布。

实际训练仍会因有限步数、学习率、数值误差和正则而不同。本 Lab 的可选正则是：

$$
L_{\mathrm{total}}
= L_{\mathrm{NLL}}+\lambda\operatorname{mean}(W^2)
= L_{\mathrm{NLL}}+\frac{\lambda}{V^2}\sum_{i,j}W_{ij}^2.
$$

它额外产生梯度 $2\lambda W_{ij}/V^2$。正则、add-alpha 平滑、label smoothing 是三种不同操作，不能宣称某个 $\lambda$ 就等于某个 $\alpha$；含正则的 total objective 也不能直接和计数版纯 NLL 比较。


---

# 自测：先写答案，再看解答

## 主线理解

1. 长度为 4 的名字，按本 Lab 的边界约定，有多少条训练样本？
2. 对 `ab, aa`，为什么 $P(b\mid a)=1/3$，而不是 $1/6$？
3. 用未平滑表算名字 `a` 的概率。它不在训练列表里，概率是 0 吗？
4. 平均 NLL 的分母为什么是 6？漏掉结束边会改变什么？
5. 每格加 $\alpha$ 后，分母为什么增加 $V\alpha$？
6. 当前字符是 `b` 时，one-hot 乘 $W$ 选中哪一行？整数 `2 * W` 又是什么？
7. 为什么每次取最大概率的字符，生成结果容易重复？
8. 为什么训练更久也不能让最后字符相同的两个前缀得到不同分布？

## 公式推导

9. $n_i=0$ 且 $\alpha=0$ 时，这一行有定义吗？$\alpha>0$ 呢？
10. 为什么 `(0,1,2)` 和 `(10,11,12)` 的 softmax 相同？
11. 预测 $(0.1,0.2,0.7)$、真实类别是 `a`，写出对 logits 的单样本梯度。
12. 六个样本、零初始化时，为什么 `a` 行的梯度是 0？
13. 某神经 bigram 的训练 NLL 比同数据的未平滑计数模型明显更低，应该先相信它更强吗？

## 解答

1. 5 条，包括预测首字符和最后预测结束。
2. 条件已经固定为当前是 `a`，此时只有 3 次观测，`a → b` 占 1 次。$1/6$ 是它在全部转移里的联合比例。
3. $P(\texttt{a})=P(a\mid\texttt{.})P(\texttt{.}\mid a)=1/3$。局部转移可以组合出训练列表里没有的完整字符串。
4. 两个名字各有 3 次预测。漏掉结束边会同时改变训练数据、停止行为和 NLL 口径。
5. 一行有 $V$ 格，每格都增加 $\alpha$。
6. 选中编号 2 的行。`2 * W` 则是把整张矩阵乘 2。
7. 起点、模型和并列规则固定后，贪心路径也固定；抽样保留了其他可能。
8. 输入只有最后一个字符。要利用不同前缀，必须改变输入信息，而不是增加训练轮数。
9. 无平滑是 $0/0$，数据无法确定这一行；$\alpha>0$ 时每一项是 $1/V$。
10. 分子分母都多一个 $e^{10}$，约掉后不变。
11. $p-q=(0.1,-0.8,0.7)$。batch 平均还要计入缩放，同一行重复选中时梯度会累加。
12. 三种后继各出现一次，经验分布正好均匀，三个样本的梯度合并后抵消。
13. 先查边界、平均分母、数据和目标是否一致。同一张自由概率表上的纯训练 NLL 已由最大似然最小化，明显更低不符合这个设定。

---

# 可运行核对实验

这段代码不依赖学生 TODO，用来核对讲义中的小数据数值。在仓库根目录运行 `uv run python` 后粘贴执行，或保存为临时文件运行。

```python
import math
import torch
import torch.nn.functional as F

# 行、列顺序为 .、a、b。
C = torch.tensor([[0, 2, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.float64)
x = torch.tensor([0, 1, 2, 0, 1, 1], dtype=torch.long)
y = torch.tensor([1, 2, 0, 1, 1, 0], dtype=torch.long)
V = 3

print("uniform NLL:", math.log(V))
for alpha in (0.0, 0.1, 1.0, 10.0):
    counts = C + alpha
    P = counts / counts.sum(dim=1, keepdim=True)
    train_nll = -P[x, y].log().mean()
    # 留出示例 ba 的三条边：.→b、b→a、a→.。
    dev_nll = -P[[0, 2, 1], [2, 1, 0]].log().mean()
    print(f"alpha={alpha:4.1f} train={train_nll.item():.6f} ba={dev_nll.item():.6f}")

# one-hot 乘法和整数查行是否相同？
W_demo = torch.arange(9, dtype=torch.float64).reshape(V, V)
X = F.one_hot(x, num_classes=V).to(torch.float64)
print("one-hot == indexing:", torch.equal(X @ W_demo, W_demo[x]))

# 核对 CE = NLL，以及 d(loss)/d(logits) = p - q。
z = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64, requires_grad=True)
target = torch.tensor([1], dtype=torch.long)
loss = F.cross_entropy(z, target)
manual_nll = -F.log_softmax(z, dim=1)[0, 1]
loss.backward()
q = F.one_hot(target, num_classes=V).to(torch.float64)
expected_grad = z.detach().softmax(dim=1) - q
print("CE = NLL:", torch.allclose(loss, manual_nll), loss.item())
print("gradient = p - q:", torch.allclose(z.grad, expected_grad))
print("single gradient:", z.grad)

# 零初始化、无正则，只做一次梯度更新。
W = torch.zeros((V, V), dtype=torch.float64, requires_grad=True)
before = F.cross_entropy(W[x], y)
before.backward()
expected_W_grad = (C.sum(1, keepdim=True) * W.detach().softmax(1) - C) / len(x)
print("batch gradient matches counts:", torch.allclose(W.grad, expected_W_grad))
print("batch gradient:\n", W.grad)
with torch.no_grad():
    W -= W.grad
with torch.no_grad():
    after = F.cross_entropy(W[x], y)
print(f"one update: {before.item():.6f} -> {after.item():.6f}")

# 任意严格正概率表都能通过 W=log(P) 被 softmax 表示。
smoothed = (C + 1) / (C + 1).sum(1, keepdim=True)
print("softmax(log(P)) = P:", torch.allclose(smoothed.log().softmax(1), smoothed))

# 核对三条具体生成路径的概率。
P0 = C / C.sum(1, keepdim=True)
ab_prob = P0[0, 1] * P0[1, 2] * P0[2, 0]
aa_prob = P0[0, 1] * P0[1, 1] * P0[1, 0]
aab_prob = P0[0, 1] * P0[1, 1] * P0[1, 2] * P0[2, 0]
print("P(ab), P(aa), P(aab):", ab_prob.item(), aa_prob.item(), aab_prob.item())
```

关键输出：

```text
uniform NLL: 1.0986122886681098
alpha= 0.0 train=0.549306 ba=inf
alpha= 0.1 train=0.607472 ba=2.266352
alpha= 1.0 train=0.835106 ba=1.364782
alpha=10.0 train=1.048931 ba=1.131055
one-hot == indexing: True
CE = NLL: True 1.4076059644443804
gradient = p - q: True
batch gradient matches counts: True
one update: 1.098612 -> 1.010797
softmax(log(P)) = P: True
```

运行前先预测：目标从 `a` 改成 `b` 时梯度怎样变；整行 logits 加 100 后 loss 是否改变；$\alpha$ 很大时两组 NLL 趋向哪里。再改代码验证。

---

# 课后资料：按需补一小块

不用读完才开始 Lab。某个具体问题说不清时，再查对应资料。

- Karpathy 的 [Zero to Hero](https://karpathy.ai/zero-to-hero.html)、[第一课视频](https://www.youtube.com/watch?v=PaCmpygFfXo) 和[官方 notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb)：对照计数、采样和神经训练。官方 notebook 留到自己写完 TODO 后再看。
- Jurafsky 与 Martin 的 [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)：补链式分解、Markov 假设和平滑。教材使用词，本 Lab 使用字符。
- 《动手学深度学习》的 [softmax 回归](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html)：补分类输出、似然和交叉熵。注意不同教材的样本维、类别维排列可能不同。
- PyTorch 的 [cross_entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) 和 [broadcasting](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)：核对 logits/targets 接口、默认平均方式，以及广播从末尾对齐。

资料检索日期：2026-09-20（UTC）。正文小数据数值以“可运行核对实验”为准。
