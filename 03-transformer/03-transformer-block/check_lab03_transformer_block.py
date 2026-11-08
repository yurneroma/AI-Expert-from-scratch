"""Public shape, gradient, parameter, and causality checks for TinyGPT."""

import torch

from lab03_transformer_block import (
  TinyGPT,
  TransformerBlock,
  TransformerConfig,
  count_parameters,
)


def small_config():
  return TransformerConfig(
    vocab_size=11,
    context_length=8,
    d_model=12,
    num_heads=3,
    num_layers=2,
    ffn_dim=24,
  )


def test_block_preserves_shape_and_backpropagates():
  block = TransformerBlock(small_config())
  inputs = torch.randn(2, 4, 12)
  output = block(inputs)
  assert output.shape == inputs.shape
  output.sum().backward()
  assert all(parameter.grad is not None for parameter in block.parameters())


def test_model_logits_parameters_and_reproducible_initialization():
  torch.manual_seed(9)
  first = TinyGPT(small_config())
  torch.manual_seed(9)
  second = TinyGPT(small_config())
  assert count_parameters(first) > 0
  assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
  inputs = torch.randint(0, 11, (1, 1))
  logits = first(inputs)
  assert logits.shape == (1, 1, 11)
  logits.sum().backward()
  assert all(parameter.grad is not None for parameter in first.parameters())


def test_full_model_remains_causal_when_future_changes():
  model = TinyGPT(small_config()).eval()
  inputs = torch.randint(0, 11, (1, 5))
  changed = inputs.clone()
  changed[:, -1] = (changed[:, -1] + 1) % 11
  with torch.no_grad():
    original_logits = model(inputs)
    changed_logits = model(changed)
  assert torch.allclose(original_logits[:, :-1], changed_logits[:, :-1])
