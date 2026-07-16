"""Compare micrograd with PyTorch using fixed inputs and parameters.

Complete the TODOs in order, then remove the ``pytest.skip`` call.
Run with:

  uv run pytest -q 01-micrograd/test_torch_alignment.py
"""

import pytest

from engine import Value
from nn import MLP


torch = pytest.importorskip("torch", reason="PyTorch is required for alignment")


def test_micrograd_matches_pytorch():

  input_values = [2.0, 3.0, -1.0]
  micrograd_inputs = [Value(value) for value in input_values]
  torch_inputs = torch.tensor(input_values, dtype=torch.float64, requires_grad=True)

  micrograd_model = MLP(3, [4, 1])
  fixed_parameters = [
    0.1, -0.2, 0.3, 0.4,
    -0.5, 0.6, -0.7, 0.8,
    0.9, -1.0, 1.1, -1.2,
    -1.3, 1.4, -1.5, 1.6,
    0.2, -0.4, 0.6, -0.8, 1.0,
  ]
  for parameter, value in zip(micrograd_model.parameters(), fixed_parameters):
    parameter.data = value

  micrograd_output = micrograd_model(micrograd_inputs)[0]
  micrograd_output.backward()

  torch_model = torch.nn.Sequential(
    torch.nn.Linear(3, 4),
    torch.nn.Tanh(),
    torch.nn.Linear(4, 1),
    torch.nn.Tanh(),
  ).to(dtype=torch.float64)

  with torch.no_grad():
    for micrograd_layer, torch_layer in zip(
        micrograd_model.layers, (torch_model[0], torch_model[2])):
      torch_layer.weight.copy_(torch.tensor(
        [[weight.data for weight in neuron.w]
         for neuron in micrograd_layer.neurons],
        dtype=torch.float64,
      ))
      torch_layer.bias.copy_(torch.tensor(
        [neuron.b.data for neuron in micrograd_layer.neurons],
        dtype=torch.float64,
      ))

  torch_output = torch_model(torch_inputs)[0]
  torch_output.backward()

  assert micrograd_output.data == pytest.approx(
      torch_output.item(), abs=1e-7, rel=0.0)

  for micrograd_layer, torch_layer in zip(
      micrograd_model.layers, (torch_model[0], torch_model[2])):
    for neuron_index, neuron in enumerate(micrograd_layer.neurons):
      for input_index, weight in enumerate(neuron.w):
        assert weight.grad == pytest.approx(
            torch_layer.weight.grad[neuron_index, input_index].item(),
            abs=1e-7, rel=0.0)
      assert neuron.b.grad == pytest.approx(
          torch_layer.bias.grad[neuron_index].item(), abs=1e-7, rel=0.0)

  for input_index, micrograd_input in enumerate(micrograd_inputs):
    assert micrograd_input.grad == pytest.approx(
        torch_inputs.grad[input_index].item(), abs=1e-7, rel=0.0)
