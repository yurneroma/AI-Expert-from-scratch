---
title: "手写反向传播自学讲义"
subtitle: "把 autograd 拆开成可核对的局部规则"
lang: zh-CN
---

# 这堂课

本课对下面的计算图手写 backward：

```text
embedding -> affine -> batchnorm -> tanh -> affine -> cross entropy
```

目标不是取代 autograd，而是理解它必须完成的 bookkeeping。**本课本身就是推导练习，但不需要先把整张图推完才写代码。** 先运行 forward、建立 shape ledger，再从 loss 端一次实现一个局部规则；详细证明放在后面。

| 阶段 | 要做什么 | 完成标志 |
|---|---|---|
| [第一阶段：直觉与 forward](#intuition) | 看懂上游梯度和多路径累加 | 能画出 forward 缓存和 shape |
| [第二阶段：分段实现](#implementation) | 从 logits 反向写到 embedding | output、BN、全参数 checks 依次通过 |
| [第三阶段：对齐与故障实验](#training) | 完成 `compare_gradients` | 每个参数有误差报告，并解释一次故意失败 |
| [第四阶段：局部规则推导](#derivations) | 实现后复盘 | 能推导 CE、matmul、broadcast 和多路径规则 |

# 第一阶段 · 先建立反向记账直觉 {#intuition}

## 1. backward 是在回答什么

forward 已经算出 loss。backward 从“loss 对自己变化的敏感度”开始，逐节点问：

> 如果中间张量的某个元素变一点，loss 会变多少？这个变化继续来自哪些输入？

每个节点接收上游梯度，再把贡献发给它的输入。一个张量如果沿多条路径影响 loss，最终梯度必须把所有路径相加。

因此 backward 不是一串需要死记的矩阵公式，而是两件重复的事：

- 用局部导数传递上游梯度。
- 把广播、reshape、重复 index 造成的多条贡献正确收回原 shape。

## 2. 先读已经给出的 forward

`forward_loss` 不需要你实现。先运行并阅读缓存：

```text
emb = C[inputs]
embcat = reshape(emb)
hprebn = embcat @ W1 + b1
bnmean = mean(hprebn)
bndiff = hprebn - bnmean
bnvar = mean(bndiff**2)
bnvar_inv = (bnvar + eps)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
h = tanh(hpreact)
logits = h @ W2 + b2
loss = mean NLL
```

每个 backward 结果必须与对应 forward 张量 shape 相同：

| Tensor | Shape |
|---|---|
| `inputs` | `(B, T)` |
| `emb` | `(B, T, E)` |
| `embcat` | `(B, T*E)` |
| `hprebn` | `(B, H)` |
| `logits` | `(B, V)` |

先跑：

```bash
uv run pytest -q 02-makemore/05-backprop/check_lab05_backprop.py -k forward
```

## 3. 先决定调试顺序

从输出向输入走：

```text
dlogits
-> dW2, db2, dh
-> dhpreact
-> BatchNorm 中间梯度
-> dW1, db1, dembcat
-> demb, dC
```

第一个 mismatch 通常才是原因，后面的 mismatch 往往只是它的后果。不要先找“误差最大的参数”。

# 第二阶段 · 从 loss 端分段实现 {#implementation}

## 1. 输出层：先让最靠近 loss 的一段对齐

平均 cross entropy 的合并规则：

```python
dlogits = probs.clone()
dlogits[range(B), targets] -= 1
dlogits /= B
```

其中 `probs = softmax(logits)`。clone 避免把 forward 缓存中的概率原地改坏；`/B` 来自 batch 平均。

接着反向输出 affine 和 tanh：

```text
dW2 = h.T @ dlogits
db2 = dlogits.sum(dim=0)
dh = dlogits @ W2.T
dhpreact = dh * (1 - h**2)
```

先在 `manual_backward` 中返回这一段需要的中间键和参数键，再运行：

```bash
uv run pytest -q 02-makemore/05-backprop/check_lab05_backprop.py -k output_layer
```

检查 `logits`、`h`、`W2`、`b2`。不要等整张图写完才开始核对。

## 2. BatchNorm：逐行反转 forward 图

从 `dhpreact` 开始，先处理最后的仿射：

```text
dbngain = 按广播收回 (dhpreact * bnraw)
dbnbias = 按广播收回 dhpreact
dbnraw = dhpreact * bngain
```

然后沿 forward 的相反顺序处理：

```text
bnraw
<- bnvar_inv, bndiff
<- bnvar, bndiff2
<- bndiff
<- bnmean, hprebn
```

这里至少有两处多路径：

- `bndiff` 直接进 `bnraw`，也通过 `bndiff2 → bnvar → bnvar_inv` 影响 loss。
- `hprebn` 直接进 `bndiff`，也通过 `bnmean → bndiff` 影响 loss。

每条路径先算出贡献，再相加；覆盖赋值会丢掉其中一路。广播回原始小 shape 时要沿扩展轴求和，mean 的反向则要把上游贡献分回所有参与平均的元素。

运行：

```bash
uv run pytest -q 02-makemore/05-backprop/check_lab05_backprop.py -k batchnorm
```

检查 `hpreact`、`bnraw`、`hprebn`、`bngain`、`bnbias`。如果失败，只在这段内部从后往前找第一个错误。

## 3. 第一层与 embedding：恢复布局并累加重复下标

继续反向：

```text
dhprebn
-> dW1 = embcat.T @ dhprebn
-> db1 = dhprebn.sum(dim=0)
-> dembcat = dhprebn @ W1.T
-> demb = dembcat.reshape(B, T, E)
```

reshape 只改变索引布局，不改变值；backward 恢复原布局即可。

embedding 前向是 `emb = C[inputs]`。同一个字符可能被多个样本和位置重复选中，所以：

```python
dC = torch.zeros_like(C)
dC.index_add_(0, inputs.reshape(-1), demb.reshape(-1, E))
```

不能用普通赋值覆盖重复下标的贡献。

最后补齐 `b1`、`W1`、`C` 等所有参数，运行：

```bash
uv run pytest -q 02-makemore/05-backprop/check_lab05_backprop.py -k all_parameters
```

# 第三阶段 · 对齐、报告和故意失败 {#training}

## TODO 2 · 用 autograd 做独立验证：`compare_gradients`

autograd 只负责提供参考答案，不参与生成 manual 结果。对每个参数报告：

```text
close
exact
max_abs_error
relative_error
```

比较原则：

- `torch.equal` 要求逐元素完全一致，代数等价但归约顺序不同时不保证。
- `torch.allclose` 使用脚手架给定的严格 `atol`、`rtol`。
- 最大绝对误差和相对误差帮助判断差异是浮点归约还是规则错误。
- 每次重新运行 autograd 前清空旧梯度，避免与 stale gradient 比较。

运行完整检查：

```bash
uv run pytest -q 02-makemore/05-backprop/check_lab05_backprop.py
```

## 必做故障实验

故意删掉一个 broadcast reduction 或把 `index_add_` 改成覆盖赋值，先在 `notes.md` 预测：

1. 会在哪个 shape 或数值上失败？
2. 哪个离 loss 最近的中间梯度仍正确？
3. 哪个梯度是第一个错误点？

运行并记录后恢复实现。这个实验比“最后一次全绿”更能说明你是否理解 reduction 和 accumulation。

## 常见失败

| 现象 | 先查什么 |
|---|---|
| 输出层梯度整体差 B 倍 | mean cross entropy 是否漏 `/B` |
| 改概率后 forward 结果异常 | 是否忘了 clone 就原地修改 |
| bias 或 gain 梯度 shape 不对 | broadcast backward 是否漏求和 |
| BN 梯度偏小 | 是否覆盖了一条多路径贡献 |
| embedding 只有少数下标正确 | 重复 index 是否覆盖而非累加 |
| 所有参数都差一点点 | 是否比较了旧 autograd 梯度，或方差约定不一致 |

# 第四阶段 · 局部规则推导（实现后复盘） {#derivations}

## 1. 反向链式法则与多路径

若 $z=f(x,y)$，上游梯度为 $\partial L/\partial z$，则：

$$
\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial z}\frac{\partial z}{\partial x},
\qquad
\frac{\partial L}{\partial y}
=\frac{\partial L}{\partial z}\frac{\partial z}{\partial y}.
$$

若同一个 $x$ 通过 $z_1,z_2$ 两条路径影响 loss：

$$
\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial z_1}\frac{\partial z_1}{\partial x}
+\frac{\partial L}{\partial z_2}\frac{\partial z_2}{\partial x}.
$$

这就是所有“gradient accumulation”的共同来源。

## 2. softmax + mean cross entropy

单个样本真实类别为 $y$：

$$
\ell=-\log p_y,
\qquad
\frac{\partial\ell}{\partial z_j}=p_j-\mathbf1[j=y].
$$

batch 损失 $L=\frac1B\sum_m\ell_m$，所以第 $m$ 行还要乘 $1/B$。这正是 `dlogits = probs - one_hot(target)` 后再 `/B` 的来源。

## 3. 矩阵乘法与 shape 对账

对 $Z=XW$：

$$
dX=dZ\,W^T,
\qquad
dW=X^T\,dZ.
$$

与其只背公式，同时检查：

```text
X:    (M, K)
W:    (K, N)
dZ:   (M, N)
dX:   (M, N) @ (N, K) -> (M, K)
dW:   (K, M) @ (M, N) -> (K, N)
```

矩阵规则把每个输出元素与所有相关输入元素的贡献一次性求和。

## 4. broadcast addition 与 mean

若 `Z = X + b`，`X` 为 `(B,H)`，`b` 为 `(H,)`，forward 把同一 `b[h]` 用在 B 个样本上，因此：

$$
dX=dZ,
\qquad
db=\sum_{\text{batch}}dZ.
$$

backward 求和是在收回广播制造的多个使用路径。

对 `m = mean(x)`，每个 $x_i$ 都以系数 $1/N$ 影响 $m$：

$$
\frac{\partial m}{\partial x_i}=\frac1N,
\qquad
dx_i=\frac{dm}{N}.
$$

这解释了 BN 中 mean 分支为什么不是简单复制上游梯度。

## 5. tanh 为什么复用 forward 输出

若 $h=\tanh(x)$：

$$
\frac{dh}{dx}=1-h^2,
\qquad
dx=dh\,(1-h^2).
$$

直接复用缓存中的 `h`，保证反向使用的正是本次 forward 的激活值。

## 6. reshape 与 embedding

reshape 没有改变数值，只改变下标映射，所以 backward 用逆 reshape 恢复 `(B,T,E)`。

embedding 中，一行 $C_i$ 可被多个位置选中：

$$
dC_i=\sum_{(b,t):X_{b,t}=i}demb_{b,t}.
$$

这和直方图累加相似：每个出现位置都向同一行加一份贡献。`index_add_` 表达的正是这个求和。

## 7. BatchNorm 的多路径为什么必须分开算

`bndiff` 的两条路径是：

```text
bndiff -> bnraw -> loss
bndiff -> bndiff2 -> bnvar -> bnvar_inv -> bnraw -> loss
```

因此：

```text
dbndiff = dbndiff_direct + dbndiff_through_variance
```

`hprebn` 也同时通过 `bndiff` 和 `bnmean` 影响 loss。完整手写时，逐个 primitive 使用前面的链式、广播和 mean 规则即可；所谓“BN 梯度公式”只是这些局部贡献合并后的结果。

## 自测

1. 哪些操作需要 backward reduction？它们为什么对应 forward 的广播或平均？
2. 本图哪些张量有超过一条 outgoing path？
3. 为什么 embedding backward 像直方图累加？
4. 为什么第一个 mismatch 比最大的 mismatch 更有定位价值？
5. forward cache 至少要保存哪些值，才能避免重算或读到被修改的张量？
