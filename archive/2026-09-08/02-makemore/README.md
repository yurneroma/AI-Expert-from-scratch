# 02 - Makemore

> 用字符级名字生成任务，从计数模型逐步走到神经网络语言模型。

这个目录是一套课程式实验，不是参考实现。每个 Lab 遵循同一个循环：

```text
read slides -> answer pre-lab questions -> implement TODOs
            -> run staged checks -> run experiment -> write notes -> commit
```

## 快速开始

```bash
# 1. 验证环境和脚手架
uv run pytest -q 02-makemore/checks/check_scaffold.py

# 2. 下载完整 names 数据（可选，已内置小数据集）
uv run python 02-makemore/scripts/download_names.py

# 3. 阅读第一讲和 Lab 01
open 02-makemore/slides/01_bigram_language_model.md
open 02-makemore/labs/LAB01_BIGRAM.md

# 4. 运行 Lab 01 检查（未实现前失败是预期行为）
uv run pytest -q 02-makemore/checks/check_lab01_bigram.py

# 5. checks 全绿后跑完整实验并生成 metrics.json
uv run python 02-makemore/experiments/run_lab01.py
```

macOS 以外的环境可以直接用编辑器打开 Markdown 文件。

## 课程地图

| Lab | 主题 | 核心产出 | 建议时间 |
|---|---|---|---:|
| 01 | Bigram 计数模型 + 单层神经网络 | NLL、平滑、采样、两种建模方式对齐 | 1-2 天 |
| 02 | Bengio MLP 语言模型 | context window、embedding、mini-batch、data split | 2 天 |
| 03 | 初始化与训练诊断 | tanh 饱和、logits scale、激活/梯度/更新统计 | 1 天 |
| 04 | BatchNorm from scratch | train/eval、running statistics、组件化深层 MLP | 1-2 天 |
| 05 | Backprop Ninja | 不借助 autograd 手写完整 backward | 2 天 |
| 06 | WaveNet-style hierarchy | receptive field、FlattenConsecutive、层次聚合 | 2 天 |
| 07 | Capstone | 统一比较、ablation、报告与博客素材 | 1 天 |

## 目录结构

```text
02-makemore/
├── slides/          # 先读：分页式讲义
├── labs/            # 再做：实验任务与验收标准
├── makemore_lab/    # 学生代码，核心函数保留 TODO
├── checks/          # 公开检查，按 Lab 单独运行
├── experiments/     # 可复现实验入口
├── notes/           # 实验记录模板
├── data/            # 离线小数据集 + 本地完整数据
├── artifacts/       # 曲线、直方图、样本、表格
└── scripts/         # 数据和工具脚本
```

## 学习约束

1. 先写张量形状，再写代码。
2. 先跑小数据和公开检查，再跑完整训练。
3. 固定随机种子；所有对比实验只改一个变量。
4. 每个 Lab 至少保存一个定量结果和一条失败观察。
5. 提交前完成对应 `notes/labXX.md`，不只提交代码。

## 常用命令

```bash
# 脚手架健康检查
uv run pytest -q 02-makemore/checks/check_scaffold.py

# 单个 Lab
uv run pytest -q 02-makemore/checks/check_lab02_mlp.py

# 只跑某个里程碑
uv run pytest -q 02-makemore/checks/check_lab02_mlp.py -k dataset

# Lab 01 完整实验（可用 --help 查看参数）
uv run python 02-makemore/experiments/run_lab01.py

# 代码语法检查
uv run python -m compileall -q 02-makemore/makemore_lab
```

`checks/` 中使用 `check_*.py` 而不是 `test_*.py`，所以仓库根目录的默认 `pytest` 不会因未完成的新 Lab 变红。
