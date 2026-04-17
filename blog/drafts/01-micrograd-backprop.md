# 反向传播的本质：从零实现 autograd

> 本文记录我从零实现 micrograd 的过程，深度理解自动微分和反向传播

## 为什么要从零实现？

在学习深度学习时，我们经常直接使用 PyTorch 或 TensorFlow，调用 `loss.backward()` 就能自动计算梯度。但这背后到底发生了什么？

通过从零实现一个支持自动微分的引擎，我想搞清楚：
1. 反向传播的数学本质是什么？
2. 计算图是如何构建的？
3. 梯度是如何自动计算的？

## Day 1: 手推梯度，建立直觉

### 从一个简单例子开始

考虑函数: `f(x, y) = (x + y) * (x - y)`

手动计算梯度：
```
∂f/∂x = ?
∂f/∂y = ?
```

[待补充：手推过程]

### 实现最小版 Value 类

```python
class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0.0
    
    def __add__(self, other):
        out = Value(self.data + other.data)
        return out
```

[待补充：详细实现]

## Day 2: 实现反向传播

### 链式法则的代码实现

[待补充]

## Day 3: 激活函数和可视化

[待补充]

## Day 4-7: 构建神经网络

[待补充]

## 核心收获

[待补充]

## 参考资料

- [Karpathy's micrograd](https://github.com/karpathy/micrograd)
- [The spelled-out intro to neural networks and backprop](https://www.youtube.com/watch?v=VMj-3S1tku0)

---

**写作日期**: 2026-04-17 ~ 04-23  
**代码仓库**: [GitHub](https://github.com/[your-username]/ai-mastery/tree/main/01-micrograd)
