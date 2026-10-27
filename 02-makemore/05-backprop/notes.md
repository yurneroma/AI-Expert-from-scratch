# Lab 05 Notes - Manual Backprop

## Pre-lab Derivations

- mean:
- broadcast addition:
- embedding lookup:
- softmax + cross entropy:

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

