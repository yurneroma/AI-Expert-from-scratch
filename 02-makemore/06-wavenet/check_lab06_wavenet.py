"""Public checks for Lab 06."""

import torch

from lab06_wavenet import (
  FlattenConsecutive,
  HierarchicalBlock,
  WaveNetNameModel,
  sample_wavenet,
)


def test_flatten_consecutive_preserves_order_and_batch_dimension():
  layer = FlattenConsecutive(2)
  inputs = torch.arange(1 * 4 * 3).reshape(1, 4, 3)
  output = layer(inputs)
  expected = torch.tensor([[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]])
  assert output.shape == (1, 2, 6)
  assert torch.equal(output, expected)


def test_flatten_rejects_incompatible_time_dimension():
  layer = FlattenConsecutive(2)
  with torch.no_grad():
    try:
      layer(torch.randn(2, 5, 3))
    except ValueError:
      pass
    else:
      raise AssertionError("expected ValueError when time is not divisible")


def test_block_halves_time_and_projects_channels():
  block = HierarchicalBlock(3, 7, group_size=2)
  inputs = torch.randn(4, 8, 3)
  output = block(inputs)
  assert output.shape == (4, 4, 7)
  output.sum().backward()
  assert all(parameter.grad is not None for parameter in block.parameters())


def test_model_returns_batch_by_vocabulary_logits():
  model = WaveNetNameModel(
    vocab_size=11,
    block_size=8,
    embedding_dim=4,
    hidden_dim=16,
    group_size=2,
  )
  inputs = torch.randint(0, 11, (5, 8))
  logits = model(inputs)
  assert model.block_size == 8
  assert logits.shape == (5, 11)
  logits.sum().backward()
  assert all(parameter.grad is not None for parameter in model.parameters())


def test_model_rejects_block_size_that_cannot_form_exact_hierarchy():
  try:
    WaveNetNameModel(vocab_size=5, block_size=6, group_size=2)
  except ValueError:
    pass
  else:
    raise AssertionError("expected ValueError for non-power block_size")


def test_sample_restores_training_mode_and_obeys_contract():
  model = WaveNetNameModel(
    vocab_size=2,
    block_size=4,
    embedding_dim=2,
    hidden_dim=4,
  )
  model.train()
  with torch.no_grad():
    for parameter in model.parameters():
      parameter.zero_()
    model.head.bias.copy_(torch.tensor([100.0, -100.0]))
  samples = sample_wavenet(
    model,
    {".": 0, "a": 1},
    {0: ".", 1: "a"},
    torch.Generator().manual_seed(1),
    n_samples=3,
  )
  assert samples == ["", "", ""]
  assert model.training

