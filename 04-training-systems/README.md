# 04 · 训练系统：测量与优化

复用第三课的模型、checkpoint、数据和评价流程，不另起训练框架。目标是先证明优化正确，再报告真实吞吐和资源变化。

## 统一学习路径

`直觉与测量问题 → 小工具 TODO → 数值/状态检查 → 基线与优化实验 → 公式选读`。速度结果只来自实际设备，不把 CPU/MPS 测量外推成 GPU 结论。

| Lab | 产出 | 入口 |
|---|---|---|
| 01 | 内存估算、阶段计时、context 趋势 | [任务](01-profiling/README.md) |
| 02 | token-weighted 梯度累积与精度对照 | [任务](02-training-efficiency/README.md) |
| 03 | KV cache 生命周期、logits 对齐、prefill/decode | [任务](03-kv-cache/README.md) |

```bash
uv run pytest -q 04-training-systems/check_scaffold.py
```

每个优化都保留 before/after、数值误差、计时协议和负结果。训练 tokens/s、prefill latency、decode ms/token 分开报告。
