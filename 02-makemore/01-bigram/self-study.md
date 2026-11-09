---

## title: "Bigram 自学讲义"
subtitle: "先直觉，再代码与训练，最后看推导"
lang: zh-CN

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

### 背景：为什么需要分数

TODO 3 结束时，我们已经得到了一张概率表。可是，两张不同的表可能都说“每一行加起来是 1”。要比较模型好坏，还需要一个数字回答：

> 对训练数据里真实出现过的每一条转移，模型给了它多少概率？

如果模型给真实转移的概率高，说明它更符合数据；如果给得低，说明它猜得差。这个数字就是 **negative log likelihood，简称 NLL**。

### 用一个名字先看计算对象

仍然用小词表：

```text
stoi = {".": 0, "a": 1, "b": 2}
```

名字 `ab` 加边界后是：

```text
. a b .
```

因此有三条真实边：

```text
. → a
a → b
b → .
```

假设概率表是：

| 当前 \| 下一个 | `.` | `a` | `b` |
|---|---:|---:|---:|
| `.` | 0 | 1.0 | 0 |
| `a` | 1/3 | 1/3 | 1/3 |
| `b` | 1.0 | 0 | 0 |

对第一条边 `. → a`，模型给出的概率是：

```text
P[0, 1] = 1.0
```

对第二条边 `a → b`：

```text
P[1, 2] = 1/3
```

对第三条边 `b → .`：

```text
P[2, 0] = 1.0
```

NLL 把每个概率转换成损失：

```text
-log(1.0)  = 0
-log(1/3)  = 1.098612
-log(1.0)  = 0
```

三个数的平均是：

```text
(0 + 1.098612 + 0) / 3 = 0.366204
```

概率越接近 1，损失越接近 0；概率越接近 0，损失越大。取 log 的原因是：一个名字的概率是每条转移概率相乘，而相乘会让很多小数连乘后变得极小。取 log 之后，乘法变成加法，数值更稳定，也更符合“逐条样本累加损失”的训练思想。

### 要做什么

实现：

```python
average_nll(words, probabilities, stoi)
```

它要返回：

```text
所有 boundary-aware bigram 的 -log P[current, next] 的平均值
```

不是：

```text
每个单词一个分数
```

也不是：

```text
只看正文字符，忽略开始和结束
```

### 怎么做

推荐按三步实现：

1. 对每个 `word` 查出字符编号，并在前后各加一次边界编号。
2. 把相邻编号变成 `(current, next)`，再用 `P[current, next]` 取出真实转移的概率。
3. 对所有取出的概率做：

```python
loss = -probabilities_of_edges.log().mean()
```

在小例子上，每个长度为 `L` 的名字会产生 `L + 1` 条边。`ab` 和 `aa` 一共有 6 条边，所以分母是 6。

一个容易出错的点是：不要在 Python 循环里对每条边调用一次 `.item()` 再累加。那样每条边都会离开 PyTorch 的批处理路径，速度慢。更好的做法是先把所有边收集成索引，再一次性索引、一次性取 log、一次性求平均。

### 为什么用它检查实现

如果 3 个字符的预测完全均匀：

```text
P[next | current] = 1/3
```

那么每条边的 NLL 都是：

```text
-log(1/3) = log(3)
```

平均也是：

```text
log(3) ≈ 1.098612
```

这是一个很好的 sanity check：如果你的函数返回的不是这个数，通常是把开始边、结束边或平均分母算错了。

对 `["ab", "aa"]` 的未平滑计数表，理论平均 NLL 是：

```text
0.549306
```

因为六条边的概率分别是：

```text
1, 1/3, 1, 1, 1/3, 1/3
```

把它们逐边转换成损失后：

```text
0, log(3), 0, 0, log(3), log(3)
```

总和是 `3 × log(3)`，再除以 6 条边：

```text
3 × log(3) / 6 = log(3) / 2 = 0.549306
```

运行：

```bash
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k nll
```

## TODO 5 · 从表里抽新名字：`sample_bigram`

### 背景：训练和生成是两件不同的事

TODO 4 评分时，`next` 来自数据里的真实名字；生成时没有真实名字，必须由模型自己决定下一个字符。

bigram 生成的状态只有一句话：

> 当前字符决定下一字符的分布。

所以必须从边界 `.` 开始；抽到普通字符就继续；抽到 `.` 就结束。

### 一个确定性小例子

假设词表只有 `.` 和 `a`：

```python
stoi = {".": 0, "a": 1}
itos = {0: ".", 1: "a"}
```

概率表是：

| 当前 \| 下一个 | `.` | `a` |
|---|---:|---:|
| `.` | 0 | 1 |
| `a` | 1 | 0 |

生成过程为：

```text
当前是 .，抽到 a，记录 a
当前是 a，抽到 .，结束
```

结果：

```text
"a"
```

这张表没有任何随机性，所以每个样本都一样。真实模型每行通常有多个非零概率，因此每次抽样可能得到不同名字。

### 要做什么

实现：

```python
sample_bigram(
    probabilities,
    stoi,
    itos,
    generator,
    n_samples=10,
    max_length=30,
)
```

返回 `n_samples` 个字符串。每个字符串都要满足：

1. 从 `stoi[.]` 对应的分布开始。
2. 用 `torch.multinomial(probabilities[current], 1, generator=generator)` 抽下一个编号。
3. 抽到边界就结束，边界本身不放进输出字符串。
4. 普通字符放入输出，并成为新的 `current`。
5. 达到 `max_length` 也停止，防止模型永远不结束。

### 怎么做

伪代码：

```text
results = []

重复 n_samples 次:
    current = stoi["."]
    chars = []

    while len(chars) < max_length:
        next = multinomial(P[current], num_samples=1, generator=generator)
        if next == boundary:
            break
        chars.append(itos[next])
        current = next

    results.append("".join(chars))

return results
```

### 为什么不用 argmax

`argmax` 每次都取概率最大的列。虽然简单，但它会把概率 0.6 和 0.4 的差别夸大成“永远选前者”。语言模型生成通常希望保留多样性，所以用按概率抽样。

`generator` 的作用是让随机结果可复现。同一个 `generator` 应连续生成所有样本；不要每个样本都新建并重置成同一个种子，否则几个样本可能完全重复。

### 检查截断行为

如果概率表是：

| 当前 \| 下一个 | `.` | `a` |
|---|---:|---:|
| `.` | 0 | 1 |
| `a` | 0 | 1 |

模型永远抽到 `a`，永远不会自然结束。设置 `max_length=4` 时必须返回：

```text
"aaaa"
```

这说明 `max_length` 是程序保护，不是模型学到的结束信号。

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

## 0. 先别急着写代码：这一阶段到底在做什么

计数模型已经能预测下一个字符了。神经版不会突然变得更聪明：它仍然只看当前字符，仍然学同一张条件概率表。

它真正要练习的是另一套能力：

```text
把训练数据整理成张量
  → 用权重算出预测分数
  → 用 loss 说出“错得有多严重”
  → 用 backward 找出“每个权重该往哪边挪”
  → 按小步更新权重
  → 重复，直到 loss 变小
```

这套流程就是后面 MLP、WaveNet、Transformer 都会用的训练闭环。这里的模型很小，正好适合把每一步看清楚。

本阶段的讲法遵循三步：

1. 先看一个具体名字。
2. 再把它翻译成 PyTorch 的张量形状。
3. 最后才写函数。

## 1. 计数模型已经能工作，为什么还要神经版

计数版的答案可以直接算：

```text
看每个当前字符后面接过什么
数次数
除以行总数
```

神经版没有这个捷径。它把“每个当前字符对每个下一个字符的偏好”放在权重表里，先随便初始化，再靠 loss 和梯度慢慢调整。

你可以把两种模型想成同一个班级的两个学生：

- 计数版学生直接背出了统计表。
- 神经版学生每次做题后根据错误修改自己的打分表。

对这个 bigram 任务，直接统计更简单、更准确。可是当模型变成多层网络、embedding、attention 之后，我们不再能用“数一数、除一除”得到最优参数，只能使用神经版这套训练方法。

因此，本阶段的目标不是得到更强的 bigram，而是把训练闭环写正确。

## TODO 6 · 把名字列表变成问答数据：`build_bigram_dataset`

### 6.1 从一句话开始：训练样本是什么？

神经网络训练时，每条样本都应该是一组“问题”和“标准答案”。

对 bigram 来说：

```text
问题：当前字符是什么？
答案：训练集中真实出现的下一个字符是什么？
```

TODO 6 要做的事情只有这一件：**把名字列表里的每一条相邻字符转移，都整理成问题和答案两列。**

它不训练、不算概率、不改词表。它只是在搬家：把字符串形式的训练材料，搬成 PyTorch 需要的整数张量。

### 6.2 一个名字怎样变成三条边

看名字：

```text
"ab"
```

先在首尾加边界符：

```text
. a b .
```

再看相邻字符。每次只看一对：

```text
. → a
a → b
b → .
```

这三条边就是三条训练样本：

| 样本编号 | 问题：当前字符 | 答案：下一个字符 |
|---:|---|---|
| 0 | `.` | `a` |
| 1 | `a` | `b` |
| 2 | `b` | `.` |

注意最后一条：

```text
b → .
```

这不是没用的边。它在教模型“什么时候结束”。如果丢掉它，模型只知道怎么往后写，不知道什么时候停。

### 6.3 `inputs` 和 `targets` 是同一张表的两列

假设词表是：

```python
stoi = {".": 0, "a": 1, "b": 2}
```

把上面三条边翻译成编号：

```text
inputs  = [0, 1, 2]
targets = [1, 2, 0]
```

二者的对应关系必须保持：

```text
inputs[0] = 0  表示当前是 .
targets[0] = 1 表示答案是 a

inputs[1] = 1  表示当前是 a
targets[1] = 2 表示答案是 b

inputs[2] = 2  表示当前是 b
targets[2] = 0 表示答案是 .
```

所以 `inputs` 和 `targets` 不是两份独立数据。它们是同一批边的左端和右端。

### 6.4 “flatten all bigrams” 到底是什么意思

TODO 注释里的这句话容易让人跳步：

```text
flatten all boundary-aware bigrams into two one-dimensional tensors
```

逐词翻译成具体动作：

- `boundary-aware`：每个名字首尾都要加边界符 `.`。
- `bigram`：相邻两个字符组成一条边，例如 `a → b`。
- `all`：不能只处理一个名字，要把所有名字的边都收集起来。
- `flatten`：不要按名字保存成嵌套结构；把所有边排成一长串。
- `two one-dimensional tensors`：左边端点放进 `inputs`，右边端点放进 `targets`。

也就是说，从：

```python
["ab", "aa"]
```

先得到逻辑上的边：

```text
ab: .→a, a→b, b→.
aa: .→a, a→a, a→.
```

再排成一张表：

| m | 来自哪个名字 | input | target |
|---:|---|---:|---:|
| 0 | `ab` | 0 | 1 |
| 1 | `ab` | 1 | 2 |
| 2 | `ab` | 2 | 0 |
| 3 | `aa` | 0 | 1 |
| 4 | `aa` | 1 | 1 |
| 5 | `aa` | 1 | 0 |

最终返回：

```python
inputs  = tensor([0, 1, 2, 0, 1, 1])
targets = tensor([1, 2, 0, 1, 1, 0])
```

两个张量的 shape 都是：

```text
(6,)
```

6 表示一共有 6 条训练边。

### 6.5 最容易踩的坑

第一，不能把所有名字直接连起来。

错误做法：

```text
.ab..aa.
```

这样会多出一条假边：

```text
. → .
```

它表示“上一个名字结束之后，下一个名字立刻开始”，但这不是原始数据里的相邻字符转移。

正确做法是每个名字单独处理：

```text
.ab.
.aa.
```

第二，不能只给开头加 `.`，必须首尾都加。

开头边界教模型“名字通常从哪些字符开始”；结尾边界教模型“哪些字符后面容易结束”。

第三，`inputs[m]` 和 `targets[m]` 不能错位。

如果第 `m` 个问题配到了别的答案，模型并不是变笨了，而是在认真学习一个错误任务。

### 6.6 要做什么

实现：

```python
build_bigram_dataset(words, stoi)
```

返回：

```python
(inputs, targets)
```

可以用这个伪代码检查思路：

```text
inputs = []
targets = []

for word in words:
    word_ids = 把 word 里的字符换成编号
    padded = [boundary_id] + word_ids + [boundary_id]

    for left, right in 相邻位置配对:
        inputs.append(left)
        targets.append(right)

return int64张量(inputs), int64张量(targets)
```

完成后运行：

```bash
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k dataset
```

如果你能不看讲义写出 `["ab", "aa"]` 的 6 条边，TODO 6 就已经理解了。

## TODO 7a · 用权重表算 loss：`neural_bigram_loss`

### 7a.1 先把 `weights` 想成一张打分表

TODO 7a 的输入不是概率表，而是：

```python
weights
```

它的行列语义仍然是：

```text
行：当前字符
列：候选的下一个字符
格子：一个可以学习的分数
```

假设词表是 `., a, b`，`weights` 可以画成：

| current \ next | `.` | `a` | `b` |
|---|---:|---:|---:|
| `.` | 0.0 | 1.2 | -0.5 |
| `a` | 0.4 | 0.0 | 0.9 |
| `b` | 1.1 | -0.2 | 0.3 |

读法是：

- 当前字符是 `.` 时，给下一个字符 `.`、`a`、`b` 分别打 0.0、1.2、-0.5 分。
- 分数越高，模型越偏好那个下一字符。
- 这些分数不是概率，叫做 **logits**。

logits 有两个重要特点：

1. 可以是正数、零、负数。
2. 一行的和不需要等于 1。

### 7a.2 logits 为什么要经过 softmax

模型最终需要概率。概率必须满足：

```text
每个概率不小于 0
整行加起来等于 1
```

softmax 把一行 logits 变成满足这些条件的概率：

$$
p_j
=
\frac{\exp(z_j)}
     {\sum_k \exp(z_k)}
$$

比如一行 logits 是：

```text
[0, 0, 0]
```

softmax 后：

```text
w[1/3, 1/3, 1/3]
```

如果其中 `a` 的 logit 变大：

```text
[0, 2, 0]
```

softmax 后 `a` 的概率也会变大。模型训练时改变 logits，softmax 后的概率就随之改变。

### 7a.3 一条训练边怎样得到 loss

先只看一条样本：

```text
input  = 0   # 当前是 .
target = 1   # 真实下一个是 a
```

第一步，从 `weights` 里选出第 0 行：

```text
z = weights[0]
```

第二步，softmax：

```text
p = softmax(z)
```

第三步，看模型给了正确答案多少概率：

```text
p[target] = p[1] = P(a | .)
```

第四步，用负对数：

```text
loss = -log(p[target])
```

如果模型给正确答案的概率接近 1：

```text
loss ≈ -log(1) = 0
```

如果给正确答案的概率很小：

```text
loss = -log(很小的数) = 很大的数
```

所以 loss 越小，表示模型越偏向训练集中的真实答案。

### 7a.4 batch 里发生了什么

TODO 6 返回的 `inputs` 是一维张量：

```text
inputs = [0, 1, 2, 0, 1, 1]
```

如果写：

```python
logits = weights[inputs]
```

PyTorch 会按编号取行：

```text
weights[0]  → 第 1 行
weights[1]  → 第 2 行
weights[2]  → 第 3 行
weights[0]  → 第 1 行
weights[1]  → 第 2 行
weights[1]  → 第 2 行
```

再把它们堆成一个二维张量：

```text
inputs:  (6,)
weights: (3, 3)
logits:  (6, 3)
```

形状的含义是：

```text
6 条训练边
每条边对 3 个候选字符各有一个分数
```

这就是 `weights[inputs]` 的全部作用：给 batch 中每个问题选出它自己的打分行。

它和 one-hot 乘矩阵等价，但整数索引更直接，也更快。

### 7a.5 `cross_entropy` 帮你做了什么

函数核心是：

```python
loss = torch.nn.functional.cross_entropy(logits, targets)
```

它接收：

```text
logits:  (B, V)
targets: (B,)
```

内部执行：

```text
对每一行 logits 做 softmax
找到每行的真实 target 概率
取 -log
对 B 条样本取平均
```

公式是：

$$
L
=
-\frac{1}{B}
\sum_{m=1}^{B}
\log
\operatorname{softmax}(z_m)_{y_m}
$$

其中：

- $z_m$ 是第 $m$ 条边的 logits 行。
- $y_m$ 是第 $m$ 条边的真实下一个字符编号。

不要先手动调用 softmax，再把结果交给 `cross_entropy`。那相当于做了两次概率转换，结果会错。

### 7a.6 一个可手算的检查例子

设：

```text
V = 3
weights 全部为 0
```

每一行 logits 都是：

```text
[0, 0, 0]
```

softmax 后都是：

```text
[1/3, 1/3, 1/3]
```

不管真实 target 是 `.`、`a` 还是 `b`，正确类别概率都是：

```text
1/3
```

所以每条边的 loss 都是：

```text
-log(1/3) = log(3) ≈ 1.098612
```

batch 平均后仍然是：

```text
1.098612
```

这个例子可以用来判断代码方向是否正确：全零权重时，loss 应该接近 `log(V)`，而不是 0。

### 7a.7 为什么返回值不能是 Python float

TODO 7a 返回的是张量：

```python
return loss
```

而不是：

```python
return loss.item()
```

原因是要保留计算图：

```text
weights → logits → softmax → loss
```

只有链条还在，后面才能执行：

```python
loss.backward()
```

并让 `weights.grad` 出现梯度。

`.item()` 会把张量变成普通 Python 数字。普通数字不知道自己是从哪些权重算出来的，也无法反向求导。

### 7a.8 正则项：额外惩罚过大的权重

默认时：

```python
regularization = 0.0
```

loss 就是纯交叉熵。

如果传入正数，就加一个权重平方惩罚：

$$
L_{\text{total}}
=
L_{\text{CE}}
+
\lambda \cdot \operatorname{mean}(W^2)
$$

直觉是：希望权重不要变得太极端。权重很大时，某些 logits 会很强，预测可能过于自信。

但要注意：

- 正则会改变优化目标。
- 含正则的 `final_objective` 不是纯 NLL。
- 不能直接拿它和计数模型的 NLL 比较。

### 7a.9 要做什么

实现：

```python
neural_bigram_loss(weights, inputs, targets, regularization=0.0)
```

流程很短：

```text
weights[inputs] 得到 batch logits
cross_entropy(logits, targets) 得到标量 loss
如果 regularization > 0，加上正则项
返回张量 loss
```

运行：

```bash
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k neural_loss
```

记住这句就够：**TODO 7a 只评分，不更新；它回答“当前 weights 有多差”。**

## TODO 7b · 训练：`train_neural_bigram`

### 7b.1 训练就是把“评分”变成“修改”

TODO 7a 会算 loss：

```text
当前 weights 有多差？
```

训练要回答：

```text
每个权重应该往哪个方向移动，loss 才会变低？
```

梯度下降的想法很朴素：

```text
如果 loss 沿某个权重的梯度是正的，
说明这个方向会让 loss 变大；
那就往反方向走一小步。
```

写成更新规则：

$$
W
\leftarrow
W
-
\eta \nabla_W L
$$

其中 $\eta$ 是学习率，用来控制每一步走多远。

学习率太大，loss 可能来回震荡；太小，下降会很慢。

### 7b.2 一轮训练的五步

每一轮完整地做五件事：

```text
1. 清空上一轮梯度
2. forward：用当前 weights 算 loss
3. backward：算所有 weights 的梯度
4. update：在 no_grad 里沿反梯度更新 weights
5. record：把本轮 loss 存进 history
```

对应 PyTorch 代码骨架：

```python
for _ in range(steps):
    if weights.grad is not None:
        weights.grad.zero_()

    loss = neural_bigram_loss(weights, inputs, targets, regularization)
    loss.backward()

    with torch.no_grad():
        weights -= learning_rate * weights.grad

    loss_history.append(loss.item())
```

这里的 `loss.item()` 只用于记录数字，发生在 `backward()` 之后，所以不会切断训练。

### 7b.3 为什么每轮都要清空梯度

PyTorch 默认会累加梯度：

```text
本轮梯度 = 上一轮残留梯度 + 本轮新梯度
```

如果不清空，模型看到的不是当前 loss 的方向，而是历史方向的混合。

所以每轮 forward 之前都要：

```python
weights.grad.zero_()
```

### 7b.4 为什么更新要放在 `torch.no_grad()` 里

`weights` 是需要学习的叶子张量：

```python
weights = torch.randn(..., requires_grad=True)
```

训练更新只是修改参数，不是为了这次修改再求导。因此要放在：

```python
with torch.no_grad():
    weights -= learning_rate * weights.grad
```

如果不加 `no_grad`，参数更新也会被记录进计算图。这样既浪费内存，也会让下一轮 backward 的对象不干净。

### 7b.5 为什么要在同一个张量上更新

正确写法是原地减法：

```python
weights -= learning_rate * weights.grad
```

不要在循环里写成：

```python
weights = weights - learning_rate * weights.grad
```

因为后者会让 `weights` 指向一个新张量，容易破坏“叶子参数”的身份。本 Lab 统一使用原地更新，问题最少。

### 7b.6 初始化为什么要用 generator

函数签名里有：

```python
seed=2147483647
```

先创建：

```python
generator = torch.Generator().manual_seed(seed)
```

再用它初始化 `weights`：

```python
weights = torch.randn(
    (vocab_size, vocab_size),
    generator=generator,
    requires_grad=True,
)
```

这样每次运行都得到同一组初始权重。调试时非常重要：如果 loss 不降，我们能先怀疑代码，而不是怀疑随机初始化。

### 7b.7 要做什么

实现：

```python
train_neural_bigram(
    inputs,
    targets,
    vocab_size,
    steps=100,
    learning_rate=20.0,
    regularization=0.01,
    seed=2147483647,
)
```

返回：

```python
(detached_weights, loss_history)
```

要求：

- `detached_weights.shape == (vocab_size, vocab_size)`
- `detached_weights.requires_grad == False`
- `len(loss_history) == steps`
- `loss_history` 中是每轮的 Python float

训练结束后，内部计算图和梯度不再是调用者需要的东西。返回一份不带计算图的权重，方便保存、比较和画图。

### 7b.8 为什么 loss 应该下降

初始权重随机，预测大约比较混乱。训练时每一步都在降低整批数据的交叉熵，让真实出现过的边获得更高概率。

在合理学习率下应该看到：

```text
history[-1] < history[0]
```

如果 loss 不降，按这个顺序查：

1. `inputs` 和 `targets` 是否错位。
2. 每轮是否清空旧梯度。
3. 是否在同一个 `weights` 张量上更新。
4. 学习率是否过大或过小。
5. 是否把 `.item()` 后的数字拿去 `backward()`。

运行：

```bash
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k neural_training
```

记住这句就够：**TODO 7b 是循环执行“评分、求梯度、更新”的机器。**

## 2. 跑完整实验

全部函数完成后运行：

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

到这里，先不看公式，也应当能用自己的话回答：

1. 为什么 `build_bigram_dataset` 返回两列，而不是一张二维计数表？
2. 为什么 `inputs[m]` 必须和 `targets[m]` 描述同一条边？
3. 为什么 `weights[inputs]` 能选出每个样本自己的 logits 行？
4. 为什么 `cross_entropy(logits, targets)` 就是在算真实下一个字符的 NLL？
5. 训练时为什么用真实 target，生成时却用模型刚抽出的字符？
6. 每轮训练为什么要清空梯度、backward、再在 `no_grad` 里更新？
7. 为什么训练更久也不能让 bigram 利用更长的上下文？

如果第 1–6 题说不清，回到对应 TODO 的小例子。第 7 题提醒我们：模型输入仍然只有当前字符，参数训练再久，也不会知道更早的前缀。

**如果目标是先会做，可以在这里停止阅读，去完成 Lab。** 下一阶段的推导用于回答“为什么”，不再是写代码的前置条件。

## 3. 出问题时，先回到小例子

调试不要从完整语料开始。先固定使用：

```python
words = ["ab", "aa"]
stoi = {".": 0, "a": 1, "b": 2}
```

然后核对：

```text
6 条边
inputs  = [0, 1, 2, 0, 1, 1]
targets = [1, 2, 0, 1, 1, 0]
```

这 12 个数字都对了，再扩大到完整数据。

| 现象 | 先查什么 |
|---|---|
| 行和不是 1 | `dim=1`、`keepdim=True`、分母是否确实是行总数 |
| 概率表有 NaN | 无平滑时是否出现空行 `0/0` |
| 新名字 NLL 无穷 | 某条真实边是否被未平滑表赋了零概率 |
| 每次生成都一样 | 是否用了 argmax，或每个名字都重置了随机种子 |
| `inputs`/`targets` 长度不对 | 是否漏掉首尾边界，或把多个名字连在一起产生假边 |
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
