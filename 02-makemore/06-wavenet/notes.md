# Lab 06 Notes - Hierarchical Model

## Intuition

1. 8 个字符怎样逐层合并：
2. 同一层共享哪些变换：
3. 与完整 WaveNet 的区别：

## Implementation Checkpoints

1. FlattenConsecutive shape:
2. 每层 Linear 的输入宽度：
3. 为什么不能用 unrestricted squeeze：
4. 采样前后 train/eval 状态：

## Shape Ledger

| Stage | Input shape | Output shape | Receptive field |
|---|---|---|---:|
| embedding | | | 1 |
| block 1 | | | |
| block 2 | | | |
| block 3 | | | |
| head | | | |

## Predictions

- flat MLP vs hierarchy:
- block size 8 vs 16:
- throughput:

## Results

| Model | Context | Parameters | Steps/s | Train NLL | Dev NLL |
|---|---:|---:|---:|---:|---:|
| Flat MLP | 8 | | | | |
| Hierarchical | 8 | | | | |
| Hierarchical | 16 | | | | |

## Failure Log

| Symptom | Root cause | Evidence | Fix |
|---|---|---|---|
| | | | |

## Explain Back

- 借用了 WaveNet 的什么？
- 与完整 WaveNet 的区别：

## Commit

- checks:
- commit:

