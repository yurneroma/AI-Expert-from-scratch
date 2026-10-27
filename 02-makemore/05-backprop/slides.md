# Lecture 05
## Manual Backpropagation

Tensor calculus as reverse-mode bookkeeping

---

## Goal

Given the graph:

```text
embedding -> affine -> batchnorm -> tanh -> affine -> cross entropy
```

derive every gradient manually and compare it with autograd.

The task is not to replace autograd. It is to understand exactly what autograd must do.

---

## Reverse-mode rule

For a node `z = f(x, y)` and upstream gradient `dL/dz`:

```text
dL/dx = dL/dz * dz/dx
dL/dy = dL/dz * dz/dy
```

If one tensor influences the loss through multiple paths, add all contributions.

---

## Keep a shape ledger

Before derivatives, write:

| Tensor | Shape |
|---|---|
| X | `(B, T)` |
| emb | `(B, T, E)` |
| embcat | `(B, T*E)` |
| hprebn | `(B, H)` |
| logits | `(B, V)` |

Every backward expression must return the shape of its corresponding forward tensor.

---

## Cross entropy shortcut

For mean cross entropy:

```text
probs = softmax(logits)
dlogits = probs
dlogits[range(B), target] -= 1
dlogits /= B
```

This is the combined derivative of softmax and negative log-likelihood.

It is simpler and more stable than differentiating each primitive separately.

---

## Matrix multiplication

For:

```text
Z = X @ W
```

the reverse pass is:

```text
dX = dZ @ W.T
dW = X.T @ dZ
```

Check dimensions before trusting the formula.

---

## Broadcast addition

For:

```text
Z = X + b
X.shape = (B, H)
b.shape = (H,)
```

the gradient is:

```text
dX = dZ
db = dZ.sum(dim=0)
```

Backward reverses broadcasting by summing over every expanded axis.

---

## Tanh

If `h = tanh(hpre)`:

```text
dhpre = dh * (1 - h**2)
```

Reuse the forward output `h`. This is both simpler and numerically coherent with the value used in the forward pass.

---

## BatchNorm graph

Expanded into primitives:

```text
mean = hprebn.mean(0)
diff = hprebn - mean
diff2 = diff**2
var = diff2.mean(0)
invstd = (var + eps)**-0.5
normalized = diff * invstd
hpreact = gain * normalized + bias
```

Manual backward should reverse this list line by line.

---

## BatchNorm accumulation points

`diff` affects the loss through:

1. `normalized = diff * invstd`;
2. `var = mean(diff**2)` which affects `invstd`.

Therefore:

```text
ddiff = ddiff_direct + ddiff_through_variance
```

Likewise, `hprebn` has a direct path through `diff` and an indirect path through `mean`.

---

## Reshape backward

If:

```text
embcat = emb.reshape(B, T*E)
```

then:

```text
demb = dembcat.reshape(B, T, E)
```

Reshape changes indexing, not values. Backward restores the original layout.

---

## Embedding lookup backward

Forward:

```text
emb = C[X]
```

Repeated characters select the same row of `C`, so gradients must accumulate:

```python
dC = torch.zeros_like(C)
dC.index_add_(0, X.reshape(-1), demb.reshape(-1, E))
```

Assignment would lose repeated contributions.

---

## Comparing gradients

For each tensor:

```text
exact = torch.equal(manual, autograd)
max_abs = (manual - autograd).abs().max()
relative = max_abs / (autograd.abs().max() + tiny)
```

Exact equality is pleasant but not always expected because algebraically equivalent reductions can differ in floating-point order.

Use strict tolerances and explain nonzero error.

---

## Debugging order

Start at the output and move backward:

```text
dlogits
-> dW2, db2, dh
-> dhpreact
-> BN intermediate gradients
-> dW1, db1, dembcat
-> demb, dC
```

The first mismatch is usually the cause; later mismatches are consequences.

---

## Common failure modes

- missing `/ B` in mean cross entropy;
- forgetting to clone probabilities before in-place edits;
- missing sum after broadcasting;
- overwriting instead of accumulating a multi-path gradient;
- using sample variance when forward used population variance;
- assigning embedding gradients instead of index-adding;
- comparing against stale autograd gradients.

---

## Check your understanding

1. Which operations require a reduction in backward?
2. Which tensors have more than one outgoing path?
3. Why does embedding backward resemble histogram accumulation?
4. Why is the first mismatch more informative than the largest mismatch?
5. What information must the forward cache preserve?

