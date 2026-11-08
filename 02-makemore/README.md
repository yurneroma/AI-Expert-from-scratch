# 02 · Makemore

从字符级 bigram 走到 context MLP，再用训练诊断和可选组件理解语言模型的训练闭环。进度、顺序和验收标准见 [schedule.md](schedule.md)。

## 统一学习方法

每个 Lab 都按同一条路线，不需要先通读公式：

```text
直觉与最小例子
-> 按 TODO 分段实现
-> 每段运行公开检查
-> 训练或受控实验
-> 记录结果与失败
-> 公式推导选读
```

| Lab | 主题 | 讲义 |
|---|---|---|
| 01 | 计数与神经 bigram | [self-study](01-bigram/self-study.md) |
| 02 | 固定 context、embedding、MLP | [讲义](02-mlp/slides.md) |
| 03 | 初始化、激活、梯度和更新诊断 | [讲义](03-diagnostics/slides.md) |
| 04（选修） | BatchNorm 与 train/eval 状态 | [讲义](04-batchnorm/slides.md) |
| 05（选修） | 手写 tensor backward | [讲义](05-backprop/slides.md) |
| 06（选修） | 分层 context 与 receptive field | [讲义](06-wavenet/slides.md) |
| 07 | 同口径模型比较与结课小结 | [任务](07-capstone/README.md) |

从仓库根目录先检查课程基础设施：

```bash
uv run pytest -q 02-makemore/check_scaffold.py
```

随后进入当前 Lab 的 README。完整数据在 `data/names.txt`，离线小数据在 `data/names_small.txt`；实验结果和失败记录写入各 Lab 的 `notes.md`。
