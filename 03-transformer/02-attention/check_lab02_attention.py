"""Public numerical, gradient, and causality checks for attention."""

import math

import torch

from lab02_attention import (
  CausalSelfAttention,
  merge_heads,
  scaled_dot_product_attention,
  split_heads,
)


def test_split_merge_heads_preserve_values_and_shape():
  x = torch.arange(2 * 3 * 8).reshape(2, 3, 8)
  split = split_heads(x, num_heads=2)
  assert split.shape == (2, 2, 3, 4)
  assert torch.equal(merge_heads(split), x)


def test_single_attention_matches_independent_float64_reference():
  torch.manual_seed(3)
  q = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
  k = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
  v = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
  mask = torch.tensor([[[True, False, False], [True, True, False], [True, True, True]]])
  output = scaled_dot_product_attention(q, k, v, mask)
  q_ref = q.detach().clone().requires_grad_()
  k_ref = k.detach().clone().requires_grad_()
  v_ref = v.detach().clone().requires_grad_()
  scores = q_ref @ k_ref.transpose(-2, -1) / math.sqrt(4)
  expected = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v_ref
  assert torch.allclose(output, expected)
  gradient = torch.randn_like(output)
  output.backward(gradient)
  expected.backward(gradient)
  for actual, reference in zip((q, k, v), (q_ref, k_ref, v_ref)):
    assert torch.allclose(actual.grad, reference.grad)


def test_causal_attention_isolates_future_and_keeps_singleton_shapes():
  torch.manual_seed(7)
  model = CausalSelfAttention(d_model=8, num_heads=2).double().eval()
  inputs = torch.randn(1, 4, 8, dtype=torch.float64)
  changed = inputs.clone()
  changed[:, 3] = 100
  first = model(inputs)
  second = model(changed)
  assert first.shape == (1, 4, 8)
  assert torch.allclose(first[:, :3], second[:, :3])
  assert model(inputs[:, :1]).shape == (1, 1, 8)
  first.sum().backward()
  assert all(parameter.grad is not None for parameter in model.parameters())
