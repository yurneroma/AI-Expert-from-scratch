# Lab 03 Notes · TinyGPT

## Model Diagram

```text
在这里画 token/position -> blocks -> norm -> head，并标代码入口
```

## Shape / Parameter Ledger

| Component | Shape / formula | Count |
|---|---|---:|
| token embedding | | |
| position embedding | | |
| one attention | | |
| one FFN | | |
| norms / head | | |
| total | | |

## Predictions / Checks

- B=1 / T=1:
- 修改未来 token:
- 固定 seed 重建:
- block and full-model gradients:

## Overfit Experiment

- config / seed / batch:
- initial loss:
- final loss:
- curve / command:

## Failure Log

| Symptom | Root cause | Evidence | Fix |
|---|---|---|---|

- commit:
