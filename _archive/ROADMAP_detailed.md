# AI 世界级专家养成计划 - 执行版
**开始日期**: 2026-04-17  
**目标**: 18个月核心能力 + 持续影响力建设 → 成为 Karpathy 级别的工程+研究专家

---

## 整体战略地图

### 三层目标体系

**Layer 1: 核心能力 (18个月)**
- 第一性原理理解神经网络
- 工业级训练与推理能力
- 原创研究与论文发表能力

**Layer 2: 影响力建设 (持续)**
- 技术博客与教学内容
- 开源项目与社区贡献
- 学术声誉与行业认知

**Layer 3: 世界级专家 (5-10年)**
- 顶会论文持续产出
- 开源项目广泛使用
- 行业标准制定参与

### 四个阶段路线

```
Phase 1: 基础夯实 (Month 1-3)
├─ micrograd: 自动微分引擎
├─ makemore: 语言模型演进
└─ nanoGPT: GPT-2 完整实现

Phase 2: 工程能力 (Month 4-6)
├─ 分布式训练 (DDP/FSDP)
├─ 推理优化 (量化/加速)
└─ 端到端项目

Phase 3: 研究能力 (Month 7-12)
├─ 精读50篇论文
├─ 复现3-5篇SOTA
└─ 原创研究+投稿

Phase 4: 影响力 (Month 13-18+)
├─ 开源项目维护
├─ 技术内容创作
└─ 社区领导力
```

---

## Phase 1 详细计划

### Month 1: micrograd (Week 1-4)

#### Week 1: 自动微分核心 (2026-04-17 ~ 04-23)

**目标**: 从零实现 autograd，深度理解反向传播

**Day 1 (04-17 周四): 手推梯度，建立直觉**

**上午 (3小时)**:
```
1. 手推简单计算图的梯度 (纸笔)
   例子: f(x,y) = (x+y) * (x-y)
   - 画出计算图
   - 手动计算 df/dx, df/dy
   - 理解链式法则的本质

2. 多推几个例子:
   - f = x * y + z
   - f = (x + y) * z
   - f = x^2 + y^2
   - f = tanh(x * w + b)
```

**下午 (3小时)**:
```
3. 创建项目结构
   mkdir -p learning/01_micrograd
   cd learning/01_micrograd
   touch engine.py
   touch test_engine.py
   touch README.md

4. 实现最小版 Value 类
   class Value:
       def __init__(self, data):
           self.data = data
           self.grad = 0.0
       
       def __repr__(self):
           return f"Value(data={self.data})"

5. 测试基本功能
   v = Value(2.0)
   print(v)
```

**晚上 (2小时)**:
```
6. 实现加法操作
   def __add__(self, other):
       out = Value(self.data + other.data)
       return out

7. 写单元测试
   a = Value(2.0)
   b = Value(3.0)
   c = a + b
   assert c.data == 5.0

8. 写今日总结 (README.md)
   - 理解了什么
   - 遇到了什么问题
   - 明天的目标
```

**学习资源**:
- Karpathy视频: "The spelled-out intro to neural networks and backprop" (前30分钟)
- 3Blue1Brown: "What is backpropagation really doing?" (可选)

**输出检查**:
- [ ] 能手推至少3个计算图的梯度
- [ ] Value 类能做加法
- [ ] 有单元测试
- [ ] 写了学习笔记

---

**Day 2 (04-18 周五): 实现四则运算和反向传播**

**上午 (3小时)**:
```
1. 实现乘法
   def __mul__(self, other):
       out = Value(self.data * other.data)
       return out

2. 实现减法和除法 (用加法和乘法组合)
   def __sub__(self, other):
       return self + (-other)
   
   def __neg__(self):
       return self * -1

3. 实现幂运算
   def __pow__(self, other):
       out = Value(self.data ** other)
       return out
```

**下午 (3小时)**:
```
4. 给 Value 添加反向传播字段
   class Value:
       def __init__(self, data, _children=()):
           self.data = data
           self.grad = 0.0
           self._backward = lambda: None
           self._prev = set(_children)

5. 实现加法的反向传播
   def __add__(self, other):
       out = Value(self.data + other.data, (self, other))
       
       def _backward():
           self.grad += out.grad
           other.grad += out.grad
       out._backward = _backward
       
       return out

6. 实现乘法的反向传播
   def __mul__(self, other):
       out = Value(self.data * other.data, (self, other))
       
       def _backward():
           self.grad += other.data * out.grad
           other.grad += self.data * out.grad
       out._backward = _backward
       
       return out
```

**晚上 (2小时)**:
```
7. 实现拓扑排序
   def backward(self):
       topo = []
       visited = set()
       def build_topo(v):
           if v not in visited:
               visited.add(v)
               for child in v._prev:
                   build_topo(child)
               topo.append(v)
       build_topo(self)
       
       self.grad = 1.0
       for node in reversed(topo):
           node._backward()

8. 测试完整的反向传播
   a = Value(2.0)
   b = Value(3.0)
   c = a * b + a
   c.backward()
   print(f"dc/da = {a.grad}")  # 应该是 4.0
   print(f"dc/db = {b.grad}")  # 应该是 2.0

9. 手动验证梯度是否正确
```

**输出检查**:
- [ ] 实现了 +, -, *, /, **
- [ ] 实现了 backward()
- [ ] 梯度计算正确
- [ ] 有完整测试

---

**Day 3 (04-19 周六): 激活函数和可视化**

**上午 (3小时)**:
```
1. 实现 tanh 激活函数
   def tanh(self):
       x = self.data
       t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
       out = Value(t, (self,))
       
       def _backward():
           self.grad += (1 - t**2) * out.grad
       out._backward = _backward
       
       return out

2. 实现 ReLU
   def relu(self):
       out = Value(max(0, self.data), (self,))
       
       def _backward():
           self.grad += (out.data > 0) * out.grad
       out._backward = _backward
       
       return out

3. 实现 exp
   def exp(self):
       x = self.data
       out = Value(math.exp(x), (self,))
       
       def _backward():
           self.grad += out.data * out.grad
       out._backward = _backward
       
       return out
```

**下午 (3小时)**:
```
4. 安装 graphviz
   pip install graphviz

5. 实现计算图可视化
   from graphviz import Digraph
   
   def trace(root):
       nodes, edges = set(), set()
       def build(v):
           if v not in nodes:
               nodes.add(v)
               for child in v._prev:
                   edges.add((child, v))
                   build(child)
       build(root)
       return nodes, edges
   
   def draw_dot(root):
       dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
       nodes, edges = trace(root)
       for n in nodes:
           dot.node(name=str(id(n)), 
                   label=f"data {n.data:.4f} | grad {n.grad:.4f}")
       for n1, n2 in edges:
           dot.edge(str(id(n1)), str(id(n2)))
       return dot

6. 可视化一个复杂计算图
   a = Value(2.0)
   b = Value(-3.0)
   c = Value(10.0)
   d = a * b + c
   e = d.tanh()
   e.backward()
   draw_dot(e).render('computation_graph', view=True)
```

**晚上 (2小时)**:
```
7. 写测试验证所有激活函数
8. 对比数值梯度和解析梯度
   def numerical_gradient(f, x, h=1e-5):
       return (f(x + h) - f(x - h)) / (2 * h)

9. 更新 README，添加可视化示例
```

**输出检查**:
- [ ] 实现了 tanh, relu, exp
- [ ] 能可视化计算图
- [ ] 数值梯度验证通过
- [ ] 有漂亮的可视化图

---

**Day 4 (04-20 周日): 神经网络层**

**上午 (3小时)**:
```
1. 创建 nn.py 文件

2. 实现 Neuron 类
   class Neuron:
       def __init__(self, nin):
           self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
           self.b = Value(random.uniform(-1, 1))
       
       def __call__(self, x):
           act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
           out = act.tanh()
           return out
       
       def parameters(self):
           return self.w + [self.b]

3. 测试单个神经元
   n = Neuron(2)
   x = [Value(1.0), Value(2.0)]
   y = n(x)
   print(y)
```

**下午 (3小时)**:
```
4. 实现 Layer 类
   class Layer:
       def __init__(self, nin, nout):
           self.neurons = [Neuron(nin) for _ in range(nout)]
       
       def __call__(self, x):
           outs = [n(x) for n in self.neurons]
           return outs[0] if len(outs) == 1 else outs
       
       def parameters(self):
           return [p for neuron in self.neurons 
                    for p in neuron.parameters()]

5. 实现 MLP 类
   class MLP:
       def __init__(self, nin, nouts):
           sz = [nin] + nouts
           self.layers = [Layer(sz[i], sz[i+1]) 
                         for i in range(len(nouts))]
       
       def __call__(self, x):
           for layer in self.layers:
               x = layer(x)
           return x
       
       def parameters(self):
           return [p for layer in self.layers 
                    for p in layer.parameters()]

6. 测试 MLP
   model = MLP(3, [4, 4, 1])
   x = [Value(1.0), Value(2.0), Value(3.0)]
   y = model(x)
   print(y)
```

**晚上 (2小时)**:
```
7. 实现参数统计
   print(f"Total parameters: {len(model.parameters())}")

8. 测试前向传播和反向传播
   y.backward()
   for p in model.parameters():
       print(p.grad)

9. 写单元测试
```

**输出检查**:
- [ ] 实现了 Neuron, Layer, MLP
- [ ] 能构建任意结构的网络
- [ ] 前向和反向传播正常
- [ ] 有完整测试

---

**Day 5 (04-21 周一): 训练循环**

**上午 (3小时)**:
```
1. 准备数据集 (XOR 或 moon)
   from sklearn.datasets import make_moons
   X, y = make_moons(n_samples=100, noise=0.1)
   y = y * 2 - 1  # 转换为 -1, 1

2. 实现损失函数
   def loss(model, X, y):
       # MSE loss
       losses = [(model(x) - yi)**2 for x, yi in zip(X, y)]
       return sum(losses) / len(losses)

3. 实现梯度清零
   def zero_grad(model):
       for p in model.parameters():
           p.grad = 0.0
```

**下午 (3小时)**:
```
4. 实现训练循环
   model = MLP(2, [16, 16, 1])
   learning_rate = 0.01
   
   for epoch in range(100):
       # Forward
       total_loss = loss(model, X, y)
       
       # Backward
       zero_grad(model)
       total_loss.backward()
       
       # Update
       for p in model.parameters():
           p.data -= learning_rate * p.grad
       
       if epoch % 10 == 0:
           print(f"Epoch {epoch}, Loss: {total_loss.data}")

5. 可视化训练过程
   import matplotlib.pyplot as plt
   
   # 记录 loss 曲线
   # 可视化决策边界
```

**晚上 (2小时)**:
```
6. 实验不同的超参数
   - learning rate: 0.001, 0.01, 0.1
   - 网络结构: [4,4,1], [16,16,1], [32,32,1]
   - 激活函数: tanh vs relu

7. 写实验报告
8. 更新 README
```

**输出检查**:
- [ ] 能训练 XOR 或 moon 数据集
- [ ] Loss 正常下降
- [ ] 有可视化结果
- [ ] 有实验对比

---

**Day 6 (04-22 周二): 优化和扩展**

**上午 (3小时)**:
```
1. 实现 mini-batch 训练
   batch_size = 32
   for epoch in range(100):
       for i in range(0, len(X), batch_size):
           batch_X = X[i:i+batch_size]
           batch_y = y[i:i+batch_size]
           # 训练

2. 实现不同的优化器
   class SGD:
       def __init__(self, parameters, lr=0.01):
           self.parameters = parameters
           self.lr = lr
       
       def step(self):
           for p in self.parameters:
               p.data -= self.lr * p.grad
       
       def zero_grad(self):
           for p in self.parameters:
               p.grad = 0.0

3. 实现 momentum
   class SGDMomentum:
       def __init__(self, parameters, lr=0.01, momentum=0.9):
           self.parameters = parameters
           self.lr = lr
           self.momentum = momentum
           self.velocities = [0.0] * len(parameters)
       
       def step(self):
           for i, p in enumerate(self.parameters):
               self.velocities[i] = (self.momentum * self.velocities[i] 
                                    - self.lr * p.grad)
               p.data += self.velocities[i]
```

**下午 (3小时)**:
```
4. 实现学习率调度
   def lr_schedule(epoch):
       if epoch < 50:
           return 0.01
       elif epoch < 100:
           return 0.001
       else:
           return 0.0001

5. 实现 early stopping
   best_loss = float('inf')
   patience = 10
   counter = 0
   
   for epoch in range(1000):
       # 训练
       if loss < best_loss:
           best_loss = loss
           counter = 0
       else:
           counter += 1
           if counter >= patience:
               break

6. 添加正则化
   def l2_reg(model, lambda_=0.01):
       reg = sum(p**2 for p in model.parameters())
       return lambda_ * reg
```

**晚上 (2小时)**:
```
7. 对比不同优化方法的效果
8. 可视化学习曲线
9. 写优化总结
```

**输出检查**:
- [ ] 实现了 SGD, Momentum
- [ ] 有学习率调度
- [ ] 有正则化
- [ ] 有性能对比

---

**Day 7 (04-23 周三): 总结和分享**

**上午 (3小时)**:
```
1. 代码重构和清理
   - 添加类型注解
   - 添加文档字符串
   - 统一代码风格

2. 写完整的 README.md
   - 项目介绍
   - 核心概念解释
   - 使用示例
   - 可视化结果

3. 添加 requirements.txt
   numpy
   matplotlib
   graphviz
   scikit-learn
```

**下午 (3小时)**:
```
4. 写技术博客: "从零实现 Autograd：理解反向传播的本质"
   
   大纲:
   - 为什么要从零实现
   - 计算图和链式法则
   - Value 类的设计
   - 反向传播的实现
   - 构建神经网络
   - 训练实例
   - 与 PyTorch 的对比
   - 学到了什么

5. 准备代码示例和可视化图
```

**晚上 (2小时)**:
```
6. 发布到 GitHub
   git init
   git add .
   git commit -m "Initial commit: micrograd implementation"
   git remote add origin <your-repo>
   git push -u origin main

7. 写推特/知乎帖子
   "花了7天从零实现了 autograd 和神经网络，
   深度理解了反向传播的本质。
   代码+博客: [链接]
   #DeepLearning #AI"

8. 回顾本周学习
   - 学到了什么
   - 遇到了什么困难
   - 下周计划 (makemore)
```

**输出检查**:
- [ ] 代码整洁，有文档
- [ ] README 完整
- [ ] 博客写完
- [ ] GitHub 发布
- [ ] 社交媒体分享

---

### Week 1 总结检查清单

**技术能力**:
- [ ] 能手推任意计算图的梯度
- [ ] 理解链式法则和反向传播
- [ ] 能从零实现 autograd
- [ ] 能构建和训练神经网络
- [ ] 理解优化器的工作原理

**代码产出**:
- [ ] engine.py (核心 autograd)
- [ ] nn.py (神经网络层)
- [ ] train.py (训练脚本)
- [ ] 完整的单元测试
- [ ] 可视化工具

**公开输出**:
- [ ] GitHub 仓库
- [ ] 技术博客
- [ ] 社交媒体分享
- [ ] 详细的 README

**理解深度**:
- [ ] 能向别人讲清楚反向传播
- [ ] 知道 PyTorch autograd 的原理
- [ ] 理解为什么需要计算图
- [ ] 知道梯度消失/爆炸的原因

---

## 后续周计划框架

### Week 2-4: makemore
- Week 2: bigram, MLP 语言模型
- Week 3: CNN (WaveNet), RNN/LSTM
- Week 4: Transformer, 总结分享

### Month 2: nanoGPT
- Week 5-6: GPT-2 架构实现
- Week 7: 训练和优化
- Week 8: 实验和总结

### Month 3: 深化理解
- Week 9-10: 精读 Attention 论文
- Week 11-12: 优化技巧实验

---

## 每周迭代流程

**周日晚上**:
1. 回顾本周完成情况
2. 写下周详细计划
3. 更新 EXECUTION_PLAN.md

**每天晚上**:
1. 写学习日志
2. 提交代码
3. 更新进度

**每周末**:
1. 写技术博客
2. 发布到社交媒体
3. 收集反馈

---

## 关键原则

1. **理解 > 完成**: 宁可慢一点，也要真正理解
2. **代码 > 理论**: 能写出来才算懂
3. **公开 > 私藏**: 边学边教，建立影响力
4. **迭代 > 完美**: 先完成，再优化
5. **聚焦 > 发散**: 一次只做一件事

---

## 下周预告

**Week 2 (04-24 ~ 04-30): makemore - bigram & MLP**
- Day 8: bigram 语言模型
- Day 9: MLP 语言模型
- Day 10: 训练和评估
- Day 11: 可视化 embedding
- Day 12: 实验不同架构
- Day 13: 优化和调参
- Day 14: 总结和分享

**准备工作**:
- 下载 names.txt 数据集
- 复习 n-gram 模型
- 了解 embedding 的概念

---

**最后提醒**: 

这不是一个"学完就结束"的计划，而是一个"持续迭代"的过程。

每周都要:
1. 写代码
2. 写博客
3. 发分享
4. 收反馈
5. 调计划

18个月后，你不仅有深厚的技术能力，还有可见的影响力。

**现在就开始 Day 1！**
