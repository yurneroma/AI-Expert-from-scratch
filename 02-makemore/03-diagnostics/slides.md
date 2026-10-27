# Lecture 03
## Initialization and Training Diagnostics

Look inside the network before trusting the loss curve

---

## A suspicious initial loss

Before training, a classifier over `V` classes should be roughly uniform:

```text
expected initial loss approximately log(V)
```

If random output logits are large, softmax becomes confident in arbitrary classes. The loss can be much larger than the uniform baseline.

The network spends early steps undoing bad confidence rather than learning structure.

---

## Control output logits

For the final layer:

```text
logits = h @ W2 + b2
```

Initialize `W2` and `b2` near zero so initial logits are near zero.

Do not initialize all hidden weights to zero: hidden neurons would remain symmetric and learn the same feature.

---

## Tanh saturation

```text
tanh'(x) = 1 - tanh(x)^2
```

When `|tanh(x)|` is near 1, the derivative is near 0.

A practical statistic:

```text
saturation_fraction = mean(|h| > 0.97)
```

A high fraction means many units transmit very little gradient through the nonlinearity.

---

## Why variance changes

For independent zero-mean inputs and weights:

```text
Var(sum_i x_i w_i) approximately fan_in * Var(x) * Var(w)
```

To keep variance at a similar scale across layers, choose:

```text
std(w) proportional to 1 / sqrt(fan_in)
```

The gain depends on the nonlinearity; `5/3` is a common tanh gain.

---

## Measure distributions, not anecdotes

For each activation tensor, inspect:

- mean;
- standard deviation;
- minimum and maximum;
- histogram;
- saturation fraction for tanh.

One printed scalar cannot show whether most values are healthy with a few outliers, or the whole layer is collapsed.

---

## Gradient diagnostics

For each parameter or retained activation gradient, inspect:

```text
mean
std
max absolute value
fraction exactly zero
```

Compare layers. A single absolute threshold is less informative than a large imbalance between early and late layers.

---

## Update-to-data ratio

For parameter `p` and SGD update `-lr * p.grad`:

```text
ratio = std(lr * p.grad) / std(p.data)
log_ratio = log10(ratio)
```

Interpretation:

- extremely small: learning is slow relative to parameter scale;
- extremely large: updates can destroy learned structure;
- useful values depend on architecture and training stage.

It is a diagnostic, not a universal law.

---

## The controlled experiment

When comparing initializations, hold constant:

- random seed;
- data split and batches;
- architecture;
- optimizer and learning rate;
- training steps.

Change only initialization scale. Otherwise you cannot attribute the result.

---

## Failure patterns

| Observation | Likely cause |
|---|---|
| initial loss far above `log(V)` | output logits too large |
| tanh mass near -1 and +1 | hidden preactivations too large |
| early-layer gradients near zero | saturation or depth |
| loss explodes after several steps | update scale too large |
| activations all near zero | weights too small or dead pathway |

These are hypotheses to test, not automatic diagnoses.

---

## PyTorch leaf-tensor trap

This expression creates a non-leaf tensor:

```python
p = torch.randn(shape, requires_grad=True) * scale
```

Safer patterns:

```python
p = torch.randn(shape) * scale
p.requires_grad_()
```

or use `torch.nn.Parameter` inside an `nn.Module`.

---

## A diagnostic loop

```text
establish expected baseline
-> inspect forward distributions
-> inspect gradients
-> inspect update scale
-> change one cause
-> rerun with same seed
```

Do not start with arbitrary hyperparameter sweeps. First identify which stage has the wrong scale.

---

## Check your understanding

1. Why can a model with a falling loss still have unhealthy saturation?
2. Why should the final layer often use a smaller initial scale?
3. What statistic distinguishes a large gradient from a large update?
4. Why is `1/sqrt(fan_in)` connected to variance preservation?
5. Which conclusions require comparison across layers rather than one tensor?

---

## Reading

- [PyTorch initialization documentation](https://pytorch.org/docs/stable/nn.init.html)
- Glorot & Bengio, Understanding the difficulty of training deep feedforward neural networks
- He et al., Delving Deep into Rectifiers

