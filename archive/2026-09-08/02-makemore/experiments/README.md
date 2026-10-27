# Experiment Runners

`run_lab01.py` 是完整的可复现实验驱动器。后续 Lab 会涉及更多你自己选择的超参数和 ablation，因此请在完成时以它为模板创建 `run_lab02.py` 到 `run_lab06.py`。

每个 runner 都应：

- 通过 CLI 显式接收 seed 和主要超参数；
- 不要把本地绝对路径写进代码；
- 将机器可读结果写入 `artifacts/labXX/metrics.json`；
- 同时打印一份人类可读摘要；
- 保存配置、数据集路径和样本，保证可复现。

