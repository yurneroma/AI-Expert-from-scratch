# Lab 05 Notes - Manual Backprop

## Forward Map

- 计算图与缓存：
- 多路径张量：
- 第一段落要核对的中间梯度：

## Implementation Rules

- mean 的反向规则：
- broadcast addition 的 reduction：
- embedding lookup 的重复 index 累加：
- softmax + cross entropy 的合并规则：

## Backward Shape Ledger

| Gradient | Shape | Reduction / accumulation required? |
|---|---|---|
| dlogits | | |
| dW2 | | |
| db2 | | |
| dhpreact | | |
| dbnraw | | |
| dhprebn | | |
| dW1 | | |
| db1 | | |
| dC | | |

## Gradient Comparison

| Parameter | Exact | Max abs error | Relative error |
|---|---:|---:|---:|
| C | | | |
| W1 | | | |
| b1 | | | |
| bngain | | | |
| bnbias | | | |
| W2 | | | |
| b2 | | | |

## Deliberate Failure

- removed reduction/accumulation:
- predicted failure:
- observed failure:
- explanation:

## Explain Back

- 三处需要 reduction/accumulation 的地方：
- 为什么应先找最靠近 loss 的 mismatch？

## Commit

- checks:
- commit:

