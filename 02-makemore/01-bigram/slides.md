> 复习提纲。第一次学习请先读 [Bigram 自学讲义](self-study.md)（[浏览器阅读版](self-study.html)），其中补齐了背景、推导、手算例子与自测。

# Lecture 01
## Bigram Language Models

From counts to differentiable learning

---

## Learning goals

After this lecture, you should be able to:

- turn a collection of words into next-character training examples;
- estimate `P(next | current)` from counts;
- evaluate a language model with negative log-likelihood;
- sample autoregressively from a probability table;
- explain why a one-layer neural bigram model solves the same task.

---

## The modeling problem

Given a prefix, predict a probability distribution over the next character.

For a bigram model, the prefix is exactly one character:

```text
P(x_t | x_1, ..., x_{t-1})  approximated by  P(x_t | x_{t-1})
```

This is a deliberately weak assumption. Its value is that every part of the language-modeling pipeline becomes inspectable.

---

## Boundary tokens are data

For the word `emma`, introduce a special token `.`:

```text
(. -> e), (e -> m), (m -> m), (m -> a), (a -> .)
```

`.` means:

- start of a word when used as the current character;
- end of a word when used as the next character.

Without the final pair, the model never learns when to stop.

---

## Vocabulary contract

Use one deterministic mapping:

```text
stoi["."] = 0
stoi["a"] = 1
...
itos[0] = "."
```

For vocabulary size `V`, the count matrix is:

```text
N.shape == (V, V)
N[i, j] == number of times token j follows token i
```

Before implementing, say what one row and one column mean in words.

---

## Maximum-likelihood estimate

For each current character `i`, normalize its row:

```text
P[i, j] = N[i, j] / sum_j N[i, j]
```

Each row is a categorical distribution:

```text
P[i, :].sum() == 1
```

The model has no parameter sharing: evidence for `a -> n` says nothing directly about `e -> n`.

---

## Smoothing

An unseen transition has count zero and therefore probability zero.

One such event makes the sequence likelihood zero and the NLL infinite.

Additive smoothing replaces:

```text
N       with       N + alpha
```

- small `alpha`: trust observed counts;
- large `alpha`: move every row toward uniform;
- `alpha = 1`: add-one smoothing.

---

## Likelihood and log-likelihood

For observed pairs `(x_i, y_i)`:

```text
likelihood = product_i P[x_i, y_i]
log_likelihood = sum_i log P[x_i, y_i]
NLL = -mean_i log P[x_i, y_i]
```

We use logs because products of many probabilities underflow, while sums of log-probabilities are stable and easy to optimize.

Lower NLL is better.

---

## A useful baseline

If the model assigns every one of `V` tokens probability `1/V`:

```text
NLL = -log(1 / V) = log(V)
```

For 27 tokens:

```text
log(27) is approximately 3.296
```

An initial loss far above this value means the initial distribution is confidently wrong, not merely untrained.

---

## Autoregressive sampling

Sampling is a loop:

```text
current = boundary token
repeat:
  next ~ Categorical(P[current])
  stop if next is boundary
  append next
  current = next
```

Use a seeded `torch.Generator` so samples are reproducible.

A maximum length is still required as a safety bound.

---

## Neural bigram model

Represent the current token as a one-hot row vector `x` and learn `W`:

```text
x.shape       = (B, V)
W.shape       = (V, V)
logits        = x @ W
logits.shape  = (B, V)
```

Because one-hot `x` selects one row, `x @ W` is equivalent to `W[x_index]`.

---

## From logits to probabilities

```text
counts = exp(logits)
probs = counts / counts.sum(dim=1, keepdim=True)
```

In production code, use `cross_entropy` or `log_softmax` for numerical stability.

The loss teaches the selected row of `W` to place more mass on the observed next token.

---

## Count model versus neural model

| Count bigram | Neural bigram |
|---|---|
| parameters are row probabilities | parameters are logits |
| fitted by counting and normalization | fitted by gradient descent |
| smoothing is explicit pseudocounts | regularization can shrink logits |
| no autograd required | differentiable training pipeline |

The neural form is not automatically more powerful here. It is preparation for architectures where rows share learned structure.

---

## Common failure modes

- forgetting start/end pairs;
- normalizing the entire matrix instead of each row;
- indexing `P[target, input]` instead of `P[input, target]`;
- taking `log(0)` without smoothing;
- using `argmax` during sampling, producing repetitive deterministic output;
- detaching logits before the loss;
- forgetting to clear gradients between optimization steps.

---

## Check your understanding

1. How many training pairs does a word of length `L` create?
2. Why does a large smoothing value approach the uniform baseline?
3. What row of `W` receives a gradient for one one-hot example?
4. Why can the count and neural models reach similar NLL?
5. What information can no bigram model represent?

---

## Reading

- [Karpathy makemore repository](https://github.com/karpathy/makemore)
- [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)

Do not read a finished Lab 01 implementation before completing your own staged checks.
