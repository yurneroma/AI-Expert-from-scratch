"""Public checks for Lab 04."""

import torch

from lab04_batchnorm import BatchNorm1d, BatchNormMLP


def test_construct_registers_parameters_and_buffers():
  layer = BatchNorm1d(3)
  parameters = dict(layer.named_parameters())
  buffers = dict(layer.named_buffers())
  assert set(parameters) == {"gamma", "beta"}
  assert set(buffers) == {"running_mean", "running_var"}
  assert parameters["gamma"].shape == parameters["beta"].shape == (3,)
  assert buffers["running_mean"].shape == buffers["running_var"].shape == (3,)


def test_training_output_has_zero_mean_and_unit_population_variance():
  layer = BatchNorm1d(3)
  layer.train()
  inputs = torch.tensor([
    [1.0, 2.0, 3.0],
    [3.0, 6.0, 9.0],
    [5.0, 10.0, 15.0],
    [7.0, 14.0, 21.0],
  ])
  output = layer(inputs)
  assert torch.allclose(output.mean(dim=0), torch.zeros(3), atol=1e-6)
  assert torch.allclose(output.var(dim=0, unbiased=False), torch.ones(3), atol=2e-5)
  assert not torch.equal(layer.running_mean, torch.zeros(3))


def test_eval_uses_running_statistics_without_updating_them():
  layer = BatchNorm1d(2)
  layer.running_mean.copy_(torch.tensor([1.0, -1.0]))
  layer.running_var.copy_(torch.tensor([4.0, 9.0]))
  layer.gamma.data.copy_(torch.tensor([2.0, 3.0]))
  layer.beta.data.copy_(torch.tensor([0.5, -0.5]))
  layer.eval()
  inputs = torch.tensor([[3.0, 2.0]])
  mean_before = layer.running_mean.clone()
  var_before = layer.running_var.clone()
  output = layer(inputs)
  expected = (
    (inputs - mean_before) / torch.sqrt(var_before + layer.eps)
  ) * layer.gamma + layer.beta
  assert torch.allclose(output, expected)
  assert torch.equal(layer.running_mean, mean_before)
  assert torch.equal(layer.running_var, var_before)


def test_gradient_reaches_input_gamma_and_beta():
  layer = BatchNorm1d(3)
  inputs = torch.randn(8, 3, requires_grad=True)
  weights = torch.tensor([1.0, 2.0, 3.0])
  (layer(inputs) * weights).sum().backward()
  assert inputs.grad is not None
  assert layer.gamma.grad is not None
  assert layer.beta.grad is not None


def test_three_dimensional_input_normalizes_batch_and_time_axes():
  layer = BatchNorm1d(4)
  inputs = torch.randn(5, 3, 4)
  output = layer(inputs)
  assert output.shape == inputs.shape
  assert torch.allclose(output.mean(dim=(0, 1)), torch.zeros(4), atol=1e-6)
  assert torch.allclose(
    output.var(dim=(0, 1), unbiased=False), torch.ones(4), atol=1e-4
  )


def test_mlp_returns_logits_and_registers_all_parameters():
  model = BatchNormMLP(
    vocab_size=7,
    block_size=3,
    embedding_dim=4,
    hidden_dims=[12, 8],
  )
  inputs = torch.randint(0, 7, (5, 3))
  logits = model(inputs)
  assert logits.shape == (5, 7)
  logits.sum().backward()
  assert all(parameter.grad is not None for parameter in model.parameters())
