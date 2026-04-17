# AI 世界级专家养成计划 - 路线图

**开始日期**: 2026-04-17  
**目标**: 18个月核心能力 + 持续影响力建设 → Karpathy 级别专家

---

## 整体战略

### 三层目标
- **Layer 1 (18个月)**: 核心能力 - 第一性原理理解 + 工程能力 + 研究能力
- **Layer 2 (持续)**: 影响力建设 - 博客 + 开源 + 社区
- **Layer 3 (5-10年)**: 世界级专家 - 顶会论文 + 广泛使用的项目 + 行业标准

### 四个阶段
```
Phase 1 (Month 1-3)  : 基础夯实 - micrograd, makemore, nanoGPT
Phase 2 (Month 4-6)  : 工程能力 - 分布式训练, 推理优化
Phase 3 (Month 7-12) : 研究能力 - 论文复现, 原创研究
Phase 4 (Month 13-18): 影响力   - 开源项目, 技术内容
```

---

## Phase 1: 基础夯实 (2026-04-17 ~ 2026-07-17)

### Month 1: micrograd + makemore (Week 1-4)

#### Week 1: micrograd - 自动微分核心 (04-17 ~ 04-23)

| 日期 | 星期 | 主题 | 核心任务 | 输出 |
|------|------|------|----------|------|
| 04-17 | 四 | 手推梯度 + Value 类 | 手推3个计算图梯度<br>实现 Value 类<br>实现加法操作 | engine.py (基础版)<br>test_engine.py<br>学习笔记 |
| 04-18 | 五 | 四则运算 + 反向传播 | 实现 *, -, /, **<br>实现 backward()<br>拓扑排序 | engine.py (完整版)<br>梯度验证通过 |
| 04-19 | 六 | 激活函数 + 可视化 | 实现 tanh, relu, exp<br>计算图可视化<br>数值梯度验证 | 可视化工具<br>漂亮的计算图 |
| 04-20 | 日 | 神经网络层 | 实现 Neuron, Layer, MLP<br>参数管理<br>前向/反向传播测试 | nn.py<br>完整测试 |
| 04-21 | 一 | 训练循环 | 准备数据集<br>实现损失函数<br>训练 moon 数据集 | train.py<br>loss 曲线<br>决策边界可视化 |
| 04-22 | 二 | 优化和扩展 | mini-batch<br>SGD + Momentum<br>学习率调度 | 优化器实现<br>性能对比 |
| 04-23 | 三 | 总结和分享 | 代码重构<br>写技术博客<br>GitHub 发布 | 完整 README<br>技术博客<br>社交媒体分享 |

**Week 1 目标**：
- 深度理解反向传播原理
- 从零实现 autograd 引擎
- 能训练简单神经网络
- 发布第一篇技术博客

**详细指导**: 见 [01-micrograd/GUIDE.md](01-micrograd/GUIDE.md)

---

#### Week 2-4: makemore - 语言模型演进 (04-24 ~ 05-14)

| 周 | 日期 | 主题 | 核心项目 | 输出 |
|----|------|------|----------|------|
| Week 2 | 04-24 ~ 04-30 | bigram + MLP | bigram 语言模型<br>MLP 语言模型 | 02-makemore/bigram.py<br>02-makemore/mlp.py<br>博客草稿 |
| Week 3 | 05-01 ~ 05-07 | CNN + RNN | WaveNet 风格 CNN<br>RNN/LSTM 实现 | 02-makemore/cnn.py<br>02-makemore/rnn.py |
| Week 4 | 05-08 ~ 05-14 | Transformer | 从零实现 Transformer<br>对比所有模型 | 02-makemore/transformer.py<br>技术博客<br>性能对比报告 |

**Month 1 里程碑**：
- ✅ 理解神经网络本质
- ✅ 掌握语言模型演进史
- ✅ 发布 2 篇技术博客
- ✅ GitHub 有 2 个完整项目

---

### Month 2: nanoGPT (Week 5-8)

| 周 | 日期 | 主题 | 核心任务 |
|----|------|------|----------|
| Week 5 | 05-15 ~ 05-21 | GPT-2 架构 | 从零实现 GPT-2 (124M)<br>理解每一行代码 |
| Week 6 | 05-22 ~ 05-28 | 训练优化 | 在 OpenWebText 训练<br>Flash Attention 集成 |
| Week 7 | 05-29 ~ 06-04 | 实验调优 | 超参数实验<br>scaling laws 验证 |
| Week 8 | 06-05 ~ 06-11 | 总结分享 | 技术博客<br>开源发布<br>中文 GPT-2 训练 |

---

### Month 3: 深化理解 (Week 9-12)

| 周 | 日期 | 主题 | 核心任务 |
|----|------|------|----------|
| Week 9 | 06-12 ~ 06-18 | Attention 论文 | 精读 "Attention Is All You Need"<br>复现关键实验 |
| Week 10 | 06-19 ~ 06-25 | Scaling Laws | 精读 scaling laws 论文<br>实验验证 |
| Week 11 | 06-26 ~ 07-02 | 优化技巧 | 实验各种优化技巧<br>性能对比 |
| Week 12 | 07-03 ~ 07-09 | Phase 1 总结 | 写总结报告<br>规划 Phase 2 |

**Phase 1 里程碑**：
- ✅ 从零实现 GPT-2
- ✅ 深度理解 Transformer
- ✅ 发布 6+ 篇技术博客
- ✅ GitHub 有 3 个核心项目
- ✅ 建立初步影响力

---

## Phase 2: 工程能力 (Month 4-6)

### 概览

| Month | 主题 | 核心项目 |
|-------|------|----------|
| Month 4 | 分布式训练 | DDP, FSDP, 混合精度 |
| Month 5 | 推理优化 | 量化, 剪枝, 推理加速 |
| Month 6 | 端到端项目 | 完整的训练到部署流程 |

**详细计划**: 待 Phase 1 完成后补充

---

## Phase 3: 研究能力 (Month 7-12)

### 概览

| 阶段 | 时间 | 核心任务 |
|------|------|----------|
| 论文精读 | Month 7-8 | 精读 50 篇顶会论文 |
| 论文复现 | Month 9-10 | 复现 3-5 篇 SOTA |
| 原创研究 | Month 11-12 | 提出新方法, 投稿顶会 |

**详细计划**: 待 Phase 2 完成后补充

---

## Phase 4: 影响力建设 (Month 13-18+)

### 概览

| 方向 | 核心任务 |
|------|----------|
| 开源项目 | 维护高质量项目, 目标 1000+ stars |
| 技术内容 | 持续写博客/视频, 建立个人品牌 |
| 社区贡献 | 参与开源社区, 帮助他人 |

**详细计划**: 待 Phase 3 完成后补充

---

## 每周迭代流程

### 周日晚上
1. 回顾本周 PROGRESS.md
2. 规划下周详细计划
3. 更新 ROADMAP.md

### 每天晚上
1. 更新 PROGRESS.md 打勾
2. 写学习笔记
3. 提交代码到 GitHub

### 每周末
1. 写技术博客
2. 发布到社交媒体
3. 收集反馈

---

## 关键原则

1. **理解 > 完成**: 宁可慢一点，也要真正理解
2. **代码 > 理论**: 能写出来才算懂
3. **公开 > 私藏**: 边学边教，建立影响力
4. **深度 > 广度**: 把核心概念打穿

---

## 文档导航

- **README.md**: 仓库首页
- **ROADMAP.md**: 本文件，整体路线图
- **PROGRESS.md**: 每日进度追踪
- **01-micrograd/GUIDE.md**: Week 1 详细指导
- **02-makemore/GUIDE.md**: Week 2-4 详细指导
- **03-nanogpt/GUIDE.md**: Week 5-8 详细指导

---

**Last Updated**: 2026-04-17  
**Current**: Phase 1, Month 1, Week 1, Day 1
