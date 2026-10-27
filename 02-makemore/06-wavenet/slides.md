# Lecture 06
## Hierarchical Context Models

Borrowing WaveNet's growing receptive field for character modeling

---

## The flat MLP bottleneck

For block size `T` and embedding width `E`, a flat MLP begins with:

```text
(B, T, E) -> (B, T*E) -> Linear(T*E, H)
```

As `T` grows, the first layer grows linearly and treats every context position in one monolithic transformation.

---

## Hierarchical grouping

Group adjacent positions repeatedly:

```text
(B, 8, E)
-> flatten pairs -> (B, 4, 2E)
-> transform     -> (B, 4, H)
-> flatten pairs -> (B, 2, 2H)
-> transform     -> (B, 2, H)
-> flatten pairs -> (B, 1, 2H)
-> transform     -> (B, 1, H)
```

Each stage doubles the receptive field.

---

## FlattenConsecutive

For group size `n`:

```text
x.shape = (B, T, C)
assert T % n == 0
y.shape = (B, T/n, C*n)
```

When `T/n == 1`, squeezing only the time axis can produce `(B, C*n)` for the final classifier.

Never use an unrestricted `squeeze()`: batch size 1 would also disappear.

---

## Receptive field

After one pairwise grouping stage, a feature summarizes 2 input characters.

After two stages, it summarizes 4.

After `L` stages:

```text
receptive_field = 2**L
```

for group size 2 and no overlap.

---

## Parameter sharing

The same transformation is applied at every group position within a level.

This is a useful inductive bias:

- local patterns can be recognized regardless of exact position;
- intermediate representations are reused;
- longer contexts do not require one independent weight block per absolute position.

---

## Relation to WaveNet

The original WaveNet models raw audio autoregressively using causal, dilated convolutions. Dilation expands the receptive field efficiently while preserving temporal resolution.

This Lab borrows:

- causal autoregressive prediction;
- exponentially growing receptive field;
- hierarchical processing of local context.

It does not reproduce the full gated residual audio architecture.

---

## A practical block

```text
FlattenConsecutive(2)
-> Linear(2 * input_channels, hidden_channels)
-> BatchNorm
-> tanh
```

Repeat until the time dimension becomes 1, then map the final hidden vector to vocabulary logits.

---

## Shape-driven construction

For block size 8 and group size 2:

```text
number_of_levels = log2(8) = 3
```

Require block size to be an exact power of the group size in the basic Lab. Handling padding or uneven trees is an extension.

---

## Fair comparison

When comparing with a flat MLP, report:

- parameter count;
- context length;
- training steps and batch size;
- train/dev NLL;
- wall-clock throughput;
- fixed-seed samples.

Lower loss from dramatically more parameters is not evidence that hierarchy itself helped.

---

## Sampling remains autoregressive

The architecture changes the function that maps context to logits. The generation loop does not change:

```text
take last block_size tokens
-> predict distribution
-> sample one token
-> append token
-> repeat
```

The model is still one-step next-token prediction.

---

## Common failure modes

- block size not divisible by group size at every stage;
- losing the batch dimension with `squeeze()`;
- giving Linear the pre-flatten channel width;
- normalizing across the feature dimension in custom BN;
- changing context ordering during reshape;
- reporting a larger model as a pure architecture improvement;
- using future characters in a supposedly causal input.

---

## Check your understanding

1. Trace every shape for `(B, 8, 10)` with hidden width 64.
2. How many original characters influence one representation at each level?
3. What does the hierarchy share that the flat first layer does not?
4. Which parts of original WaveNet are absent here?
5. Why does autoregressive sampling stay unchanged?

---

## Reading

- [van den Oord et al., WaveNet (2016)](https://arxiv.org/abs/1609.03499)

Focus on causality, dilation, and receptive-field growth. Treat the paper as design context, not as code to copy.

