# micrograd

> 从零实现自动微分引擎，深度理解反向传播

## 项目目标

通过从零构建一个支持自动微分的神经网络库，深度理解：
1. 反向传播的数学原理
2. 自动微分的实现机制
3. 神经网络的本质

## 项目结构

```
01-micrograd/
├── README.md           # 本文件
├── engine.py           # 核心: Value 类和自动微分
├── nn.py               # 神经网络层 (Neuron, Layer, MLP)
├── test_engine.py      # 单元测试
├── demo.ipynb          # 可视化演示
├── train_moon.py       # 训练示例
└── blog_draft.md       # 技术博客草稿
```

## 核心概念

### 1. Value 类
封装标量值和梯度，支持自动微分

### 2. 计算图
记录操作历史，支持反向传播

### 3. 拓扑排序
确保梯度按正确顺序传播

## 使用示例

```python
from engine import Value

# 前向传播
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
e = d.tanh()

# 反向传播
e.backward()

# 查看梯度
print(f"de/da = {a.grad}")
print(f"de/db = {b.grad}")
```

## 学习资源

- [Karpathy 视频: The spelled-out intro to neural networks and backprop](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [3Blue1Brown: What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

## 进度

- [ ] Day 1: Value 类基础
- [ ] Day 2: 反向传播
- [ ] Day 3: 激活函数 + 可视化
- [ ] Day 4: 神经网络层
- [ ] Day 5: 训练循环
- [ ] Day 6: 实战项目
- [ ] Day 7: 总结 + 博客

## 技术博客

完成后将发布技术博客: "反向传播的本质：从零实现 autograd"

---

**开始日期**: 2026-04-17  
**预计完成**: 2026-04-23
