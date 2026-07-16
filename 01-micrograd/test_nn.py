"""Day 4 acceptance tests.

Work from top to bottom. During development, run one stage at a time:

  uv run pytest -q 01-micrograd/test_nn.py -k neuron
  uv run pytest -q 01-micrograd/test_nn.py -k layer
  uv run pytest -q 01-micrograd/test_nn.py -k mlp
"""

import math

from engine import Value
from nn import Layer, MLP, Neuron


def test_neuron_initializes_one_parameter_per_input_plus_bias():
  neuron = Neuron(3)

  assert len(neuron.w) == 3
  assert all(isinstance(weight, Value) for weight in neuron.w)
  assert isinstance(neuron.b, Value)
  assert all(-1.0 <= parameter.data <= 1.0 for parameter in neuron.parameters())


def test_neuron_forward_matches_known_calculation():
  neuron = Neuron(2)
  neuron.w = [Value(0.5), Value(-1.0)]
  neuron.b = Value(0.1)

  output = neuron([2.0, 3.0])
  expected = math.tanh(0.5 * 2.0 + (-1.0) * 3.0 + 0.1)

  assert isinstance(output, Value)
  assert math.isclose(output.data, expected, rel_tol=0.0, abs_tol=1e-12)


def test_neuron_backward_reaches_every_parameter():
  neuron = Neuron(2)
  output = neuron([Value(1.5), Value(-2.0)])

  output.backward()

  assert any(parameter.grad != 0.0 for parameter in neuron.parameters())


def test_layer_creates_requested_number_of_neurons():
  layer = Layer(3, 4)

  assert len(layer.neurons) == 4
  assert all(len(neuron.w) == 3 for neuron in layer.neurons)


def test_layer_returns_one_value_per_neuron():
  layer = Layer(3, 4)
  outputs = layer([1.0, 2.0, 3.0])

  assert isinstance(outputs, list)
  assert len(outputs) == 4
  assert all(isinstance(output, Value) for output in outputs)


def test_layer_parameters_are_flat():
  layer = Layer(3, 4)

  # Each of 4 neurons owns 3 weights and 1 bias.
  assert len(layer.parameters()) == 4 * (3 + 1)
  assert all(isinstance(parameter, Value) for parameter in layer.parameters())


def test_mlp_builds_expected_architecture():
  model = MLP(3, [4, 4, 1])

  assert len(model.layers) == 3
  assert len(model.layers[0].neurons) == 4
  assert len(model.layers[0].neurons[0].w) == 3
  assert len(model.layers[1].neurons) == 4
  assert len(model.layers[1].neurons[0].w) == 4
  assert len(model.layers[2].neurons) == 1
  assert len(model.layers[2].neurons[0].w) == 4


def test_mlp_forward_and_backward_run_end_to_end():
  model = MLP(3, [4, 4, 1])

  outputs = model([2.0, 3.0, -1.0])
  assert isinstance(outputs, list)
  assert len(outputs) == 1
  assert isinstance(outputs[0], Value)

  outputs[0].backward()
  assert any(parameter.grad != 0.0 for parameter in model.parameters())


def test_mlp_parameters_are_flat_and_complete():
  model = MLP(3, [4, 4, 1])

  # Layer 1: 4 * (3 weights + 1 bias) = 16
  # Layer 2: 4 * (4 weights + 1 bias) = 20
  # Layer 3: 1 * (4 weights + 1 bias) = 5
  assert len(model.parameters()) == 16 + 20 + 5
  assert all(isinstance(parameter, Value) for parameter in model.parameters())
