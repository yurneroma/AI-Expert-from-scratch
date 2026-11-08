---
title: "初始化与训练诊断自学讲义"
subtitle: "先测量，再判断训练为什么有问题"
lang: zh-CN
---

# 这堂课

上一课让 MLP 跑通了。本课不再只问“loss 有没有下降”，而是问：**初始预测、隐藏激活、梯度和参数更新的尺度是否健康？**

第一次学习先建立诊断直觉，再实现六个小工具，最后做受控初始化实验。公式推导放在实验之后。

| 阶段 | 要做什么 | 完成标志 |
|---|---|---|
| [第一阶段：直觉](#intuition) | 看懂 loss 曲线之外的三类信号 | 能提出一个可检验的故障假设 |
| [第二阶段：实现诊断工具](#implementation) | 完成 TODO 1–6 | 统计、梯度和绘图 checks 通过 |
| [第三阶段：受控实验](#training) | 比较三组初始化 | 只改变初始化，能用分布和比率解释结果 |
| [第四阶段：公式推导](#derivations) | 实验后选读 | 能解释 `log(V)`、tanh 饱和和 fan-in scaling |

# 第一阶段 · loss 下降不一定代表健康 {#intuition}

## 1. 一个可疑的开头

对 `V` 个字符，如果初始 logits 都接近 0，softmax 接近均匀，初始 cross entropy 应接近 `log(V)`。

如果输出层随机权重太大，模型会在没有任何依据时非常自信。真实字符可能只拿到很小概率，初始 loss 就远高于 `log(V)`。训练最初一段时间只是在撤销这种任意自信，而不是学习语言规律。

所以看到异常初始 loss，先检查输出 logits，不要立刻扫学习率。

## 2. 中间层可能“卡住”

隐藏层使用 tanh。当输出接近 `-1` 或 `1` 时，曲线已经变平，梯度接近 0。即使总 loss 还在缓慢下降，大量隐藏单元也可能已经饱和，后续学习会很慢。

一个 loss 标量无法区分：

- 大部分激活健康，只有少量离群值；
- 整层激活都接近饱和；
- 激活全部接近 0，几乎没有有效信号。

因此要看分布、直方图和饱和比例。

## 3. 梯度大，不代表更新相对参数一定大

SGD 更新由 `learning_rate * gradient` 决定。参数本身很大时，同样的梯度可能只带来很小的相对改变；参数很小时，同样更新可能把结构冲坏。

本课因此分别测量：

```text
forward：logits 和激活分布
backward：各层梯度分布
update：更新尺度相对于参数尺度
```

这些指标用于提出和验证假设，没有脱离架构与训练阶段的万能阈值。

# 第二阶段 · 先写诊断工具 {#implementation}

## TODO 1 · 权重尺度：`recommended_weight_std`

本 Lab 使用规则：

```text
std = gain / sqrt(fan_in)
```

`fan_in` 是一个输出单元接收的输入数量。它必须为正。`gain` 根据非线性调整，tanh 常用 `5/3`。

现在先把它当作控制初始激活尺度的规则；“为什么是平方根”留到[后置推导](#variance-derivation)。

运行：

```bash
uv run pytest -q 02-makemore/03-diagnostics/check_lab03_diagnostics.py -k weight_std
```

公开检查：`fan_in=25, gain=1` 时结果为 `0.2`；`gain=5/3` 时为 `1/3`。

## TODO 2 · 描述一个张量：`tensor_statistics`

返回普通 Python floats：

```text
mean、std、min、max、zero_fraction
```

约定：

- 使用总体标准差，即 `unbiased=False`。
- 测量前 detach，不把诊断图表接进训练图。
- 明确定义空张量行为，不能等到 NaN 才发现没有数据。

运行：

```bash
uv run pytest -q 02-makemore/03-diagnostics/check_lab03_diagnostics.py -k tensor
```

对 `[-1, 0, 1, 0]`，均值是 0，总体标准差是 `sqrt(0.5)`，零值比例是 `0.5`。

## TODO 3 · 测量 tanh 饱和：`tanh_saturation_fraction`

统计：

```python
(activations.abs() > threshold)
```

所占比例。本 Lab 默认阈值为 `0.97`，并要求 `0 <= threshold < 1`。正负两端都算饱和，所以要取绝对值。

运行：

```bash
uv run pytest -q 02-makemore/03-diagnostics/check_lab03_diagnostics.py -k saturation
```

## TODO 4 · 分层看梯度：`gradient_statistics`

对每个已有 `.grad` 的参数复用 `tensor_statistics`。如果请求的参数没有梯度，抛出清晰错误，不要静默跳过。

比较有价值的通常不是“某层梯度是否小于某个固定数”，而是：

- 相近层之间是否差几个数量级；
- 是否大量梯度恰好为 0；
- 前层与后层是否严重失衡；
- 改变初始化后，同一 seed 下分布怎样变化。

运行：

```bash
uv run pytest -q 02-makemore/03-diagnostics/check_lab03_diagnostics.py -k gradient
```

## TODO 5 · 比较更新与参数：`update_to_data_ratio`

对参数 `p` 和学习率 `lr`，本 Lab 返回：

```text
log10(std(lr * p.grad) / (std(p.data) + eps))
```

两者都使用总体标准差，并要求已有梯度。

- 比率极小：更新相对于参数很慢。
- 比率极大：更新可能破坏已学结构。
- 具体合理范围依赖模型和训练阶段，应比较趋势与层间差异。

公开检查中，`p=[-1,1]`、`grad=[-0.1,0.1]`、`lr=0.1` 时，结果为 `-2`。

运行：

```bash
uv run pytest -q 02-makemore/03-diagnostics/check_lab03_diagnostics.py -k update
```

## TODO 6 · 保存分布图：`save_distribution_plot`

把多个张量 detach 到 CPU，画带标签的重叠密度直方图，保存后关闭 figure。图像用于观察整体分布和离群值，不替代前面的数值表。

运行：

```bash
uv run pytest -q 02-makemore/03-diagnostics/check_lab03_diagnostics.py
```

# 第三阶段 · 用受控实验回答原因 {#training}

## 1. 只改变初始化

复用 Lab 02 的同一个 MLP，比较：

| Variant | Hidden weight scale | Output weight scale |
|---|---:|---:|
| Bad | `1.0` | `1.0` |
| Controlled | `5/3 / sqrt(fan_in)` | `0.01` |
| Too small | `0.01 / sqrt(fan_in)` | `0.001` |

保持完全一致：

- 数据 split 和 batch 顺序；
- 网络结构；
- optimizer、学习率、训练步数；
- 随机 seed 和评估方式。

否则无法把差异归因于初始化。

## 2. 按 forward → backward → update 收集证据

每组保存：

1. step 0 的 logits 分布与初始 loss。
2. tanh 激活分布、直方图和饱和比例。
3. 训练中期的各层梯度标准差和零梯度比例。
4. 各层 update/data ratio。
5. train/dev loss 曲线。

在 `notes.md` 先写预测，再写结果。诊断顺序固定为：

```text
建立均匀 loss 基线
-> 看 forward 分布
-> 看梯度
-> 看更新尺度
-> 只改变一个原因
-> 用同一 seed 重跑
```

## 3. 结果怎样变成结论

| 观察 | 可检验的假设 | 下一项证据 |
|---|---|---|
| 初始 loss 远高于 `log(V)` | 输出 logits 过大 | step 0 logits 分布 |
| tanh 大量接近 ±1 | 隐藏预激活过大 | 饱和比例与梯度 |
| 前层梯度接近 0 | 饱和或深层传播问题 | 分层梯度对照 |
| 几步后 loss 爆炸 | 更新尺度过大 | update/data ratio |
| 激活全接近 0 | 权重过小或路径失活 | 激活 std 与梯度 |

“loss 正常下降但大量 tanh 饱和”就是本课要寻找的典型反例：一个标量改善了，内部训练动力学仍可能不健康。

# 第四阶段 · 公式推导（实验后选读） {#derivations}

## 1. 为什么均匀分类的初始 loss 是 `log(V)` {#initial-loss-derivation}

若 $V$ 个类别每个概率都是 $1/V$，真实类别的 NLL 为：

$$
-\log(1/V)=\log V.
$$

每个样本都相同，平均后仍为 $\log V$。若 logits 很大且真实类别不是模型偏爱的类别，$p_y$ 很小，$-\log p_y$ 就会高于这个基线。

## 2. tanh 饱和为什么让梯度变小

令 $h=\tanh(x)$：

$$
\frac{dh}{dx}=1-h^2.
$$

当 $|h|\to1$，$1-h^2\to0$。上游梯度传到预激活 $x$ 时还要乘这一项，因此饱和单元向后传递的梯度很小。

阈值 `0.97` 只是本 Lab 的测量约定，不是数学上的绝对分界。

## 3. 为什么权重标准差随 `1/sqrt(fan_in)` 缩放 {#variance-derivation}

考虑一个预激活：

$$
z=\sum_{i=1}^{n}x_iw_i.
$$

在零均值、输入和权重相互独立、各项同分布的简化假设下：

$$
\operatorname{Var}(z)
\approx n\,\operatorname{Var}(x)\operatorname{Var}(w).
$$

若希望 $\operatorname{Var}(z)$ 与 $\operatorname{Var}(x)$ 接近，就让：

$$
n\operatorname{Var}(w)\approx1
\quad\Rightarrow\quad
\operatorname{std}(w)\propto\frac1{\sqrt n}.
$$

这里的 $n$ 就是 fan-in。真实网络中的相关性、非线性和训练过程会破坏这些简化假设，所以它是初始化设计原则，不是逐层方差永远不变的保证。

## 4. 为什么比较 update/data，而不是只看梯度

一次 SGD 更新为 $\Delta p=-\eta\,p.grad$。参数尺度是 $\operatorname{std}(p)$，更新尺度可用 $\operatorname{std}(\eta\,p.grad)$ 表示，因此：

$$
r=\frac{\operatorname{std}(\eta\,p.grad)}
{\operatorname{std}(p)+\varepsilon},
\qquad
\log_{10}r
$$

把相对尺度放到便于观察数量级的坐标上。它不直接告诉你 loss 会升还是降，只提示更新相对于参数是否过小或过大。

## 自测

1. 为什么 loss 下降仍可能伴随严重 tanh 饱和？
2. 为什么最终输出层常用更小初始化，而隐藏层不能全部置零？
3. 哪个指标区分“梯度大”和“相对参数的更新大”？
4. 为什么比较初始化时必须固定 batch 顺序和 seed？
5. 哪些结论需要跨层比较，不能只看一个张量？

## 阅读

- [PyTorch initialization 文档](https://pytorch.org/docs/stable/nn.init.html)
- Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*
- He et al., *Delving Deep into Rectifiers*

先对照文档理解 fan-in 和 gain；论文用于完成实验后的深入阅读。
