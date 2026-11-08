"""Public RMSNorm numerical, gradient, zero-input, and switch checks."""

import pytest
import torch
from torch import nn

from lab02_component_change import RMSNorm, build_norm


def test_rmsnorm_forward_and_gradients_match_independent_float64_reference():
  torch.manual_seed(13)
  layer = RMSNorm(4).double()
  inputs = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
  reference_inputs = inputs.detach().clone().requires_grad_()
  reference_gain = layer.gain.detach().clone().requires_grad_()
  actual = layer(inputs)
  expected = reference_inputs * torch.rsqrt(
    reference_inputs.square().mean(dim=-1, keepdim=True) + layer.eps
  ) * reference_gain
  assert torch.allclose(actual, expected)
  gradient = torch.randn_like(actual)
  actual.backward(gradient)
  expected.backward(gradient)
  assert torch.allclose(inputs.grad, reference_inputs.grad)
  assert torch.allclose(layer.gain.grad, reference_gain.grad)


def test_rmsnorm_zero_and_small_variance_inputs_are_finite():
  layer = RMSNorm(3)
  assert torch.equal(layer(torch.zeros(2, 3)), torch.zeros(2, 3))
  output = layer(torch.full((1, 2, 3), 1e-8))
  assert torch.isfinite(output).all()


def test_norm_switch_preserves_baseline_choice_and_rejects_unknown():
  baseline = build_norm("layer_norm", 4)
  variant = build_norm("rmsnorm", 4)
  assert isinstance(baseline, nn.LayerNorm)
  assert isinstance(variant, RMSNorm)
  with pytest.raises(ValueError, match="kind"):
    build_norm("unknown", 4)
