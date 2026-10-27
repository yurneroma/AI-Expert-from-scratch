# 02 · Makemore

先修：micrograd 已完成。目标：掌握 tensor、next-token prediction、NLL、数据划分和 mini-batch 训练，接上 Transformer。

预算：W1-2（2026-09-08 至 09-21），约 32 小时任务 + 8 小时缓冲。每次按本表选一个 Lab，在该目录内完成阅读、实现、校验和记录。只更新本表状态和完成日期。

| 周 / 预算 | Lab | 阅读 | 验收产出 | 状态 / 完成日期 |
|---|---|---|---|---|
| W1 · 10h | [01 · Bigram](01-bigram/README.md) | [自学讲义](01-bigram/self-study.md) · [复习 slides](01-bigram/slides.md) | 边界 token、计数/神经 bigram、NLL 与采样；8 项检查 | 下一步 / — |
| W1-2 · 12h | [02 · Context MLP](02-mlp/README.md) | [slides](02-mlp/slides.md) | word split、embedding、训练/验证与采样；7 项检查 | 待做 / — |
| W2 · 8h | [03 · 训练诊断](03-diagnostics/README.md) | [slides](03-diagnostics/slides.md) | 初始化对照、饱和比例、梯度和更新统计；6 项检查 | 待做 / — |
| W2 · 2h | [07 · 结课小结](07-capstone/README.md) | 回看 Lab 01-03 的图和 notes | 同一 split 下的 Bigram/MLP 对照表；口头解释训练闭环 | 待做 / — |
| 选修 · 6-8h | [04 · BatchNorm](04-batchnorm/README.md) | [slides](04-batchnorm/slides.md) | train/eval 和 running statistics；6 项检查 | 选修 / — |
| 选修 · 8-12h | [05 · Tensor Backprop](05-backprop/README.md) | [slides](05-backprop/slides.md) | 完整 tensor 梯度与 autograd 对齐；4 项检查 | 选修 / — |
| 选修 · 8-12h | [06 · WaveNet](06-wavenet/README.md) | [slides](06-wavenet/slides.md) | 层级 receptive field；依赖 Lab 04；6 项检查 | 选修 / — |

选修保留原编号，方便回看既有练习。Lab 03 后只需额外口头推导 `softmax + CE`、一次 affine backward 和 embedding 重复 index 的梯度累加；完整 BN backward 可以延后。

## 开始第一份作业

环境沿用根目录 `pyproject.toml`；从仓库根目录执行：

```bash
uv run pytest -q 02-makemore/check_scaffold.py
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py -k vocab
```

第二条目前应在 TODO 1 失败。依次完成同目录的 `lab01_bigram.py`，随后运行全部公开检查和实验：

```bash
uv run pytest -q 02-makemore/01-bigram/check_lab01_bigram.py
uv run python 02-makemore/01-bigram/run_lab01.py
```

实验输出进入 `01-bigram/artifacts/metrics.json`，结果写进 [notes](01-bigram/notes.md)。完整数据在 `data/names.txt`；新环境可运行 `uv run python 02-makemore/scripts/download_names.py`，离线练习使用 `data/names_small.txt`。

所有 `check_*.py` 必须显式指定，根目录默认 `pytest` 不代表这些作业已通过。选修 Lab 的 TODO 失败不影响必修验收。

完成标准：必修检查通过；能解释每层 shape 和 NLL；在同一 word split 下比较 Bigram 与 MLP 的 dev loss；能过拟合小 batch，能排查梯度未清零或 tanh 饱和。性能结论以自己的测量为准，单次未超过 baseline 时先排查和解释。

完成后提交本 Lab 代码、notes 和实验摘要，更新本表，再进入 [03 · Transformer](../03-transformer/schedule.md)。
