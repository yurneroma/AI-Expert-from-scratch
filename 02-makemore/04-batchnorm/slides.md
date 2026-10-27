# Lecture 04
## Batch Normalization from Scratch

Normalization, learnable scale, and train/eval state

---

## The operation

For one feature over a mini-batch:

```text
mean = mean(x)
variance = mean((x - mean)^2)
x_hat = (x - mean) / sqrt(variance + eps)
y = gamma * x_hat + beta
```

`gamma` and `beta` are learned per feature.

---

## Shape contract

For 2-D input:

```text
x.shape = (B, C)
statistics axes = (0,)
```

For sequence input:

```text
x.shape = (B, T, C)
statistics axes = (0, 1)
```

The final dimension is the feature dimension and is preserved.

---

## Why gamma and beta matter

Normalization alone forces every feature to zero mean and unit variance.

The learned affine transform lets the model choose a different scale and offset, including approximately undoing normalization when useful.

```text
gamma.shape = (C,)
beta.shape = (C,)
```

Broadcasting applies them across batch and time dimensions.

---

## Training mode

During training:

1. compute statistics from the current mini-batch;
2. normalize with those statistics;
3. update running estimates without gradient tracking.

```text
running = (1 - momentum) * running + momentum * batch_stat
```

This convention must stay consistent in code and tests.

---

## Evaluation mode

At inference, a sample should not depend on which unrelated examples happen to share its batch.

Use stored running statistics:

```text
x_hat = (x - running_mean) / sqrt(running_var + eps)
```

Calling `eval()` changes behavior; it does not freeze `gamma` and `beta` by itself.

---

## State categories

| State | Learned by gradient? | Saved in state dict? |
|---|---:|---:|
| gamma | yes | yes |
| beta | yes | yes |
| running_mean | no | yes |
| running_var | no | yes |

In PyTorch, learned values are Parameters; running statistics are buffers.

---

## Linear bias redundancy

Suppose:

```text
hpre = x @ W + b
```

Batch normalization subtracts the batch mean, which includes `b`. A following learned `beta` supplies an offset again.

Therefore a linear bias immediately before BN is usually unnecessary.

---

## Batch dependence

Training-mode output for one example depends on the other examples in its mini-batch.

Consequences:

- batch size affects statistic noise;
- very small batches may behave poorly;
- BN introduces a regularizing stochastic effect;
- train and eval paths must both be tested.

---

## Variance details

Training normalization commonly uses the population-style batch variance (`unbiased=False`). Running variance conventions can differ between implementations.

For this Lab, use the contract in the scaffold and checks rather than mixing formulas from different libraries.

The goal is an internally consistent implementation.

---

## Deep MLP composition

One block:

```text
Linear(no bias) -> BatchNorm -> tanh
```

Repeat blocks, then use a final Linear layer for logits.

The final logits layer usually does not need BN because cross entropy expects unconstrained class scores.

---

## Common failure modes

- reducing over the feature axis;
- storing running statistics as trainable Parameters;
- updating running statistics while in eval mode;
- allowing running updates into the autograd graph;
- using batch statistics during sampling;
- forgetting `eps`;
- incompatible broadcasting for `(B, T, C)`.

---

## Check your understanding

1. For `(B, T, C)`, what shape do mean and variance have with `keepdim=True`?
2. Why can a single sample produce different training-mode outputs in two batches?
3. Which four tensors must be saved for inference?
4. Why is BN not just standardizing the input dataset once?
5. What exactly changes when `model.eval()` is called?

---

## Reading

- [Ioffe & Szegedy, Batch Normalization (2015)](https://arxiv.org/abs/1502.03167)
- [PyTorch BatchNorm1d documentation](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)

The original paper's explanation is historically important, but separate the proposed mechanism from later debates about why BN helps.

