# Lab 02 · 改一个组件

## Objective

默认只把 LayerNorm 换成 RMSNorm，保留 Pre-LN 位置和可切换 baseline。先做小 tensor 数值/梯度检查，再过拟合小 batch，最后接完整训练。

## Start Here

按 [slides](slides.md)：先看清[只改哪一处](slides.md#intuition)，再完成[RMSNorm 与开关](slides.md#implementation)，随后做[模型接回与故障实验](slides.md#training)，最后选读[公式和与其他组件的关系](slides.md#derivations)。不要求先推公式才写代码。

## Student File

`lab02_component_change.py`

1. `RMSNorm.forward`
2. `build_norm`

RoPE、SwiGLU、GQA 是替代选题，不与默认 RMSNorm 同时引入；对应额外检查见 slides。

## Staged Checks

```bash
uv run pytest -q 05-architecture-lab/02-component-change/check_lab02_component_change.py -k forward_and_gradients
uv run pytest -q 05-architecture-lab/02-component-change/check_lab02_component_change.py -k zero
uv run pytest -q 05-architecture-lab/02-component-change/check_lab02_component_change.py -k switch
uv run pytest -q 05-architecture-lab/02-component-change/check_lab02_component_change.py
```

## Definition of Done

- [ ] float64 forward、输入梯度、gain 梯度与独立参考对齐。
- [ ] 零输入、小方差、不同 B/T 有限且 shape 不变。
- [ ] 开关关闭时 baseline 路径不变，checkpoint 保存结构配置。
- [ ] 故意改错归一化维度时检查失败，随后恢复。
- [ ] 参数量/FLOPs/内存差异有记录，整模型仍保持因果。
