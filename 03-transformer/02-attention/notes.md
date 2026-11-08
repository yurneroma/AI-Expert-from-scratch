# Lab 02 Notes · Causal Attention

## Intuition / Mask

- Q/K/V 各自的作用：
- True/False mask 语义：
- 为什么未来不能影响过去：

## Shape Ledger

| Tensor | Shape | Meaning |
|---|---|---|
| X | | |
| Q/K/V | | |
| split heads | | |
| scores | | |
| probabilities | | |
| merged output | | |

## Predictions

- 修改未来输入后：
- T=1 时：
- softmax 维写错时：

## Checks / Failure Log

| Check / symptom | Evidence | Fix |
|---|---|---|

- numerical/gradient result:
- causal result:
- commit:
