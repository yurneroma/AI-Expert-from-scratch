# micrograd 实现指南 (合同 #1, Day 1-7)

> 规则：先自己写，卡住超过 30 分钟才看视频对应片段；**任何时候不看 Karpathy 的源码**，Day 6 才对照。

## Day 1 (06-10): 手推梯度 + Value 类

**理论 (1h)**: 在纸上手推这 3 个计算图的所有偏导，拍照存入 `notes/`：
1. `e = a*b + c`，求 de/da, de/db, de/dc
2. `f = (a+b) * (a+c)`（注意 a 出现两次——梯度累加的来源）
3. `g = tanh(a*w + b)`（tanh'(x) = 1 - tanh²(x)）

**实现 (3h+)**: `engine.py`
```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    def __add__(self, other): ...
    def __mul__(self, other): ...
```

**DoD**:
- [ ] 3 张手推照片在 repo
- [ ] `Value` 支持 `+` `*`，`_prev`/`_op` 记录正确
- [ ] `test_engine.py`: 至少 5 个测试（含 `a+a` 这种重复引用）
- [ ] commit + push

## Day 2 (06-11): backward 全链路

- 实现 `__pow__` `__neg__` `__sub__` `__truediv__`（除法 = 乘 -1 次幂）、`__radd__` `__rmul__`
- 实现 `backward()`: 拓扑排序（DFS post-order）→ 反向遍历调用 `_backward`
- 关键点: 梯度**累加**（`+=` 不是 `=`），起点 `self.grad = 1.0`

**DoD**:
- [ ] Day 1 手推的 3 个图，代码算出的梯度与手推完全一致
- [ ] 测试覆盖所有运算符
- [ ] commit + push

## Day 3 (06-12): 激活函数 + 数值梯度验证

- 实现 `tanh` `relu` `exp` `log`
- 写 `grad_check(f, inputs, eps=1e-6)`: 中心差分 `(f(x+eps) - f(x-eps)) / (2*eps)` 对比解析梯度
- 加分项: graphviz 可视化计算图

**DoD**:
- [x ] 所有 op 通过数值梯度验证（相对误差 < 1e-4）
- [x ] commit + push

## Day 4 (06-13): Neuron / Layer / MLP

`nn.py`:
```python
class Neuron:   # w·x + b → tanh
class Layer:    # n 个 Neuron
class MLP:      # 多个 Layer, e.g. MLP(3, [4, 4, 1])
```
- 实现 `parameters()` 统一收集参数
- **对齐测试**: 同一组权重/输入下，与 PyTorch 的梯度逐项对比（这是今天的核心，允许用 torch 仅做验证）

**DoD**:
- [ ] MLP 前向+反向跑通
- [ ] 与 PyTorch 梯度对齐（误差 < 1e-6）
- [ ] commit + push

## Day 5 (06-14): 训练循环 + moon 数据集

`train_moon.py`:
- sklearn `make_moons` 100 个点
- hinge loss 或 MSE + L2 正则
- 训练循环: forward → zero_grad → backward → SGD 更新
- 画 loss 曲线 + 决策边界图，存入 repo

**踩坑预告**: 忘记 zero_grad 是经典 bug，故意先不写，观察现象再修——这个体验值得写进博客。

**DoD**:
- [ ] moon 数据集准确率 > 95%
- [ ] 两张图入 repo
- [ ] commit + push

## Day 6 (06-15): 对照原版 + 重构

- 通读 [karpathy/micrograd](https://github.com/karpathy/micrograd) 源码（现在才允许）
- 写 `notes/diff-vs-karpathy.md`: ≥3 条差异分析（设计取舍，不是风格差异）
- 重构自己的代码

**DoD**:
- [ ] 差异分析 ≥3 条
- [ ] commit + push

## Day 7 (06-16): 博客 #1

《从零实现 autograd：反向传播的本质》
- 素材: 手推照片、zero_grad 踩坑、与 PyTorch 对齐的过程、与原版的差异
- 发布到公开平台（知乎/掘金/个人博客均可），链接记入 PROGRESS.md

**DoD**:
- [ ] 博客公开发布
- [ ] README 更新项目状态
- [ ] commit + push

---

**资源**（仅在卡住时使用）:
- [Karpathy: spelled-out intro to backprop](https://www.youtube.com/watch?v=VMj-3S1tku0)（2.5h，建议 Day 1 晚上先完整看一遍再开始 Day 2）
