# Day 5 Lab: Train an MLP on Two Moons

## Learning objective

Today you will turn the scalar network from Day 4 into a complete learning
system. The target pipeline is:

```text
data -> forward pass -> hinge loss -> zero gradients -> backward -> SGD
```

You may use scikit-learn to create the dataset and matplotlib to draw figures.
The model, loss graph, gradients, and parameter updates must use your own
`Value` and `MLP` implementations.

## Part 1: Dataset

Complete TODO 1 in `train_moon.py`. Generate 100 two-dimensional moon samples
and convert labels from `0/1` to `-1/+1`, because the hinge-loss formula uses
the sign of the label.

```bash
uv run pytest -q 01-micrograd/test_train_moon.py -k dataset
```

Before continuing, inspect a few `(x, y)` pairs. Why is a straight line unable
to separate the two classes?

## Part 2: Objective function

Complete TODO 2 through TODO 4. For example `i`:

```text
score_i = model(x_i)[0]
hinge_i = max(0, 1 - y_i * score_i)
```

Average the hinge terms and add L2 regularization. Build these expressions from
`Value` operations; extracting `.data` while constructing the loss disconnects
the computation graph.

```bash
uv run pytest -q 01-micrograd/test_train_moon.py -k loss
```

Questions:

1. When is a correctly classified point still penalized by hinge loss?
2. Why does L2 regularization inspect parameters rather than predictions?
3. Which operation in `engine.py` implements `max(0, x)`?

## Part 3: One optimization step

Complete TODO 5 and TODO 6. SGD minimizes the loss, so its update uses the
negative gradient direction:

```text
parameter = parameter - learning_rate * gradient
```

Run the focused checks:

```bash
uv run pytest -q 01-micrograd/test_train_moon.py -k 'zero_grad or sgd'
```

Then deliberately omit `zero_grad()` for two iterations and inspect a parameter
gradient. `Value._backward()` accumulates with `+=`, so stale gradients silently
change the optimizer step.

## Part 4: Training loop

Complete TODO 7 in this exact order for every epoch:

```text
1. forward and loss
2. zero_grad(model)
3. total_loss.backward()
4. sgd_step(model, current_learning_rate)
5. record plain numeric metrics
```

Use `current_learning_rate = learning_rate * (1 - 0.9 * epoch / epochs)`.
Each history record has this contract:

```python
{"epoch": 0, "loss": 0.83, "accuracy": 0.52}
```

Run the complete logic test:

```bash
uv run pytest -q 01-micrograd/test_train_moon.py
```

The final test is intentionally slower because this scalar engine creates a
fresh computation graph for the full dataset on every epoch.

## Part 5: Visual evidence

Complete TODO 8 through TODO 10 and run:

```bash
uv run python 01-micrograd/train_moon.py
```

Save both files under `01-micrograd/`:

- `loss_curve.png`: epoch on x-axis, loss on y-axis.
- `decision_boundary.png`: colored prediction regions plus the 100 samples.

For the decision boundary, create a dense 2-D grid, run each point through the
trained model, and classify by whether the output is greater than zero.

## Definition of done

- [ ] All ten TODOs in `train_moon.py` are implemented and removed.
- [ ] `uv run pytest -q 01-micrograd/test_train_moon.py` passes.
- [ ] `uv run pytest -q 01-micrograd` keeps earlier tests passing.
- [ ] Final moon accuracy is greater than 95%.
- [ ] `loss_curve.png` and `decision_boundary.png` are committed.
- [ ] Record the zero-gradient experiment in your blog notes.
- [ ] Commit and push Day 5.
