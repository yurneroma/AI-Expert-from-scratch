# Day 4 Lab: Build a Multilayer Perceptron

## Learning objective

You already built scalar automatic differentiation. In this lab, you will use
only `Value`, Python lists, and loops to assemble a small neural network.

By the end, this program must run:

```python
from nn import MLP

model = MLP(3, [4, 4, 1])
outputs = model([2.0, 3.0, -1.0])
outputs[0].backward()

for parameter in model.parameters():
  print(parameter.data, parameter.grad)
```

Do not use NumPy or PyTorch to implement `nn.py`. PyTorch will only be used
later as an independent reference for gradient comparison.

## Public API contract


| Component | Constructor        | Call result                    | `parameters()`     |
| --------- | ------------------ | ------------------------------ | ------------------ |
| `Neuron`  | `Neuron(nin)`      | one `Value`                    | flat `list[Value]` |
| `Layer`   | `Layer(nin, nout)` | `list[Value]` of length `nout` | flat `list[Value]` |
| `MLP`     | `MLP(nin, nouts)`  | final `list[Value]`            | flat `list[Value]` |


For this lab, `Layer` and `MLP` always return a list, even when there is only
one output. Keeping one return type makes the layer-composition logic easier to
reason about.

## Part 1: Neuron

A neuron computes:

```text
activation = w1*x1 + w2*x2 + ... + wn*xn + b
output = tanh(activation)
```

Complete TODO 1 through TODO 3 in `nn.py`.

Requirements:

1. `Neuron(nin)` creates exactly `nin` weights and one bias.
2. Every weight and bias is a `Value` initialized in `[-1, 1]`.
3. `__call__(x)` returns one `Value`.
4. Use the `Value` arithmetic you already wrote. Do not manually calculate
  gradients inside `nn.py`.
5. `parameters()` returns the weights and bias in one flat list.

Run only the neuron tests:

```bash
uv run pytest -q 01-micrograd/test_nn.py -k neuron
```

Because the Layer test names also contain the word `neuron`, pytest may select
five tests here (`3` Neuron tests plus `2` Layer-construction tests). The exact
selection is less important than making the first three Neuron tests pass.

Questions to answer before moving on:

1. Why must each weight be a `Value` instead of a plain float?
2. Where is the computation graph created during a neuron's forward pass?
3. Why does `Neuron` not need its own `backward()` method?



## Part 2: Layer

A layer contains `nout` neurons. Every neuron receives the same input vector,
but owns independent weights and a bias.

```text
             -> Neuron 1 -> output 1
input vector -> Neuron 2 -> output 2
             -> Neuron 3 -> output 3
```

Complete TODO 4 through TODO 6.

Requirements:

1. `Layer(nin, nout)` creates exactly `nout` neurons.
2. Every neuron accepts exactly `nin` inputs.
3. `__call__(x)` returns one output per neuron.
4. `parameters()` returns one flat list, not a nested list.

Run the neuron and layer tests:

```bash
uv run pytest -q 01-micrograd/test_nn.py -k 'neuron or layer'
```

Expected milestone after both components are implemented: `6 passed, 3 deselected`.

Parameter-count check:

```text
Layer(nin, nout) has nout * (nin + 1) parameters.
```

The `+1` is the bias owned by each neuron.

## Part 3: MLP

An MLP chains layers together. The output width of one layer becomes the input
width of the next layer.

```text
MLP(3, [4, 4, 1])

3 inputs -> Layer(3, 4) -> Layer(4, 4) -> Layer(4, 1) -> 1 output
```

Complete TODO 7 through TODO 9.

Requirements:

1. Build the layer sizes from `[nin] + nouts`.
2. Pair adjacent sizes to construct each `Layer`.
3. In `__call__`, feed each layer's output into the next layer.
4. `parameters()` returns every parameter from every layer in one flat list.

Run all Day 4 tests:

```bash
uv run pytest -q 01-micrograd/test_nn.py
```

Expected milestone: `9 passed`.

For `MLP(3, [4, 4, 1])`, the parameter count is:

```text
first layer:  4 * (3 + 1) = 16
second layer: 4 * (4 + 1) = 20
third layer:  1 * (4 + 1) = 5
total:                         41
```



## Debugging checklist

If a forward test fails, inspect these in order:

1. Does every neuron have the correct number of weights?
2. Does the weighted sum start from the bias?
3. Are you applying `tanh()` to the final activation?
4. Does `Layer.__call__` return a list?
5. Does `MLP.__call__` replace `x` with each layer's output?

If backward produces zero gradients:

1. Confirm weights and biases are `Value` objects.
2. Confirm the forward pass uses `Value.__add__` and `Value.__mul__`.
3. Confirm you call `backward()` on a final output `Value`.
4. Remember that a saturated `tanh` can produce a very small gradient. Small
  is not the same as disconnected.



## Definition of done

- [ ] All nine TODOs in `nn.py` are implemented.
- [ ] `uv run pytest -q 01-micrograd/test_nn.py` reports nine passing tests.
- [ ] `uv run pytest -q 01-micrograd` keeps all engine tests passing.
- [ ] `MLP(3, [4, 4, 1])` returns one final `Value` inside a list.
- [ ] Calling `backward()` gives gradients to the model parameters.

After this lab passes, the next task is an independent PyTorch gradient-alignment
test using fixed weights and inputs.