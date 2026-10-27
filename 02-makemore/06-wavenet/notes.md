# Lab 06 Notes - Hierarchical Model

## Pre-lab

1. block size 8 的层级数：
2. FlattenConsecutive shape:
3. hierarchical inductive bias:
4. parameters vs receptive field:

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

