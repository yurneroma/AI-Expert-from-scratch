# Lecture 02
## A Neural Probabilistic Language Model

Longer context, learned embeddings, and proper evaluation

---

## What bigrams cannot know

The next character often depends on more than one preceding character.

```text
P(x_t | x_{t-1})
```

cannot distinguish two prefixes that end in the same character.

Move to a fixed context window:

```text
P(x_t | x_{t-T}, ..., x_{t-1})
```

---

## Dataset construction

For block size 3 and word `emma`, start with context `[..., ..., ...]`:

```text
[., ., .] -> e
[., ., e] -> m
[., e, m] -> m
[e, m, m] -> a
[m, m, a] -> .
```

After each target, shift left and append the target.

Shapes:

```text
X.shape = (N, T)
Y.shape = (N,)
```

---

## Embedding lookup

Let:

```text
C.shape = (V, E)
X.shape = (B, T)
```

Then:

```text
emb = C[X]
emb.shape = (B, T, E)
```

Advanced indexing selects rows, but gradients still accumulate into the selected rows of `C`.

---

## Flatten the context

Concatenate all token embeddings:

```text
embcat = emb.reshape(B, T * E)
```

Hidden layer:

```text
W1.shape = (T * E, H)
b1.shape = (H,)
h = tanh(embcat @ W1 + b1)
```

Output layer:

```text
W2.shape = (H, V)
b2.shape = (V,)
logits = h @ W2 + b2
```

---

## Parameter count

Total parameters:

```text
V*E + (T*E)*H + H + H*V + V
```

Increasing context length expands the first affine layer. This motivates later hierarchical grouping.

---

## Cross entropy

For each row, cross entropy selects the target log-probability:

```text
loss = -mean(log_softmax(logits)[range(B), Y])
```

Prefer:

```python
torch.nn.functional.cross_entropy(logits, Y)
```

It fuses stable log-softmax and NLL computation.

---

## Mini-batch SGD

One step:

```text
sample batch indices
forward
compute cross entropy
clear gradients
backward
update parameters
```

Mini-batches make each gradient noisy but cheap. The loss curve should be interpreted as a noisy measurement, not a monotonic contract.

---

## Split by words, not examples

If examples are generated first and then randomly split, adjacent prefixes from the same word can appear in train and validation.

Correct order:

```text
shuffle words -> split words -> build examples independently
```

Typical proportions:

```text
80% train / 10% dev / 10% test
```

---

## Dataset roles

| Split | Purpose |
|---|---|
| train | fit parameters |
| dev | choose architecture and hyperparameters |
| test | one final unbiased report |

Repeatedly choosing settings based on test loss converts the test set into another development set.

---

## Underfit and overfit

```text
high train loss, high dev loss       -> underfit or optimization failure
low train loss, much higher dev loss -> overfit
both improve together                -> useful capacity or better optimization
```

Always compare the gap as well as the absolute dev loss.

---

## Sampling the MLP

At generation time:

```text
context = [0] * block_size
repeat:
  logits = model(context)
  next ~ softmax(logits)
  context = context[1:] + [next]
```

Training and sampling must use exactly the same token mapping and context convention.

---

## What embeddings learn

Characters that are useful in similar contexts can move toward nearby vectors.

The model learns geometry because the same embedding table is reused at every context position.

With two-dimensional embeddings, plotting `C` is a useful qualitative diagnostic, but visual appeal is not an evaluation metric.

---

## Common failure modes

- building train/dev/test examples before splitting words;
- flattening away the batch dimension;
- wrong `W1` input width after changing block size;
- creating non-leaf parameters by scaling tensors after enabling gradients;
- evaluating without `torch.no_grad()`;
- sampling from logits without softmax;
- tuning on test loss.

---

## Check your understanding

For `B=32, T=3, E=10, H=200, V=27`, write every shape from `X` through `logits`.

Then answer:

1. Which parameter grows with context length?
2. What does a row of `C` represent?
3. Why can train loss improve while dev loss worsens?
4. What would block size 1 reduce to conceptually?

---

## Reading

- [Bengio et al., A Neural Probabilistic Language Model (2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

Focus on the model diagram and distributed representation idea; the Lab uses characters instead of words.

