# Composer 2: RL for Coding Agents

**论文**: [Composer 2 Technical Report](https://arxiv.org/html/2603.24477v1)  
**作者**: Cursor Team  
**日期**: 2026-03  
**阅读日期**: 2026-04-17  
**状态**: 初步了解，待 Phase 3 精读

---

## 为什么记录这篇

在 Week 1 Day 1 学习 micrograd 时，偶然讨论到这篇论文。虽然现在还没到研究阶段，但论文中关于 reward 设计和环境对齐的思想很有启发，先记录下来。

---

## 核心思想（一句话）

**训练环境必须和部署环境完全一致，reward 设计要反映真实的用户体验，而不只是"对不对"。**

---

## 关键点速记

### 1. 环境对齐（Environment Alignment）
- 训练时用的工具、harness、后端和生产环境**完全相同**
- 维护 shadow deployment，训练和生产共享实现
- 每个训练环境跑在独立 Firecracker VM
- 目标：最小化 train-test mismatch

### 2. Reward 设计（三层）

**主 reward**：
- 代码正确性
- 代码简洁性
- 软件工程原则

**行为 reward**：
- 编码风格
- 沟通方式
- 工具调用质量
- 惩罚不良行为（创建 TODO 不完成、注释里写思维链等）

**长度惩罚（非线性）**：
```
C(x) = ((1+kx)^(1-q) - 1) / (k(1-q))
```
- 简单任务：每多一步惩罚感知强
- 困难任务：允许多想多试
- 反映真实用户心理

### 3. Reward 的真实含义

- Reward 不只是"对不对"，还要考虑用户体验
- 持续监控涌现行为，动态调整 reward
- RL 不只是重新分配概率，而是学到新的解题路径

### 4. 自我总结机制

- 长任务中模型多次自我总结
- 好的总结被强化，差的总结被弱化
- 比外部 prompt compaction 更好

---

## 对我的启发

1. **环境对齐 > 算法创新**：工程设计比算法更重要
2. **Reward 是多层信号的组合**：不是一个数字
3. **非线性惩罚反映真实偏好**：reward 函数形状要匹配人类感受
4. **持续监控涌现行为**：RL 训练会产生意想不到的行为

---

## 待深入的问题

- [ ] GRPO 算法的具体实现
- [ ] 非线性长度惩罚的参数如何调优
- [ ] 如何监控和识别涌现行为
- [ ] 自我总结的 reward 如何设计
- [ ] Anyrun 基础设施的技术细节

---

## 精读计划

**时间**: Phase 3 (Month 7-8)  
**前置知识**: 
- Transformer 架构
- RL 基础（PPO/GRPO）
- 大模型训练经验

**精读重点**:
1. Reward 设计的完整细节
2. 环境对齐的工程实现
3. 异步 RL 训练的架构
4. 实验结果和 ablation study

---

## 相关资源

- [arXiv 论文](https://arxiv.org/html/2603.24477v1)
- [Cursor 官方博客](https://cursor.com/blog/composer-2-technical-report)
- [DataCamp 解读](https://www.datacamp.com/blog/composer-2)

---

**Tags**: #RL #reward-design #environment-alignment #coding-agent #phase3-todo
