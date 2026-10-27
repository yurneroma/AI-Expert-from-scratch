"""Public checks for Lab 02."""

import pytest
import torch
import torch.nn.functional as F

from lab02_mlp import (
  build_context_dataset,
  evaluate_mlp_loss,
  initialize_parameters,
  mlp_logits,
  sample_mlp,
  split_words,
  train_mlp,
)


def test_split_is_reproducible_disjoint_and_does_not_mutate_input():
  words = [f"name{letter}" for letter in "abcdefghij"]
  original = words.copy()
  first = split_words(words, seed=7)
  second = split_words(words, seed=7)
  assert first == second
  assert words == original
  train, dev, test = first
  assert len(train) == 8 and len(dev) == 1 and len(test) == 1
  assert set(train).isdisjoint(dev)
  assert set(train).isdisjoint(test)
  assert set(dev).isdisjoint(test)


def test_dataset_shifts_context_after_every_target():
  stoi = {".": 0, "a": 1, "b": 2}
  inputs, targets = build_context_dataset(["ab"], stoi, block_size=2)
  assert inputs.dtype == targets.dtype == torch.int64
  assert torch.equal(inputs, torch.tensor([[0, 0], [0, 1], [1, 2]]))
  assert torch.equal(targets, torch.tensor([1, 2, 0]))


def test_initialize_returns_leaf_parameters_with_expected_shapes():
  parameters = initialize_parameters(5, 3, 2, 7, seed=1)
  assert set(parameters) == {"C", "W1", "b1", "W2", "b2"}
  assert parameters["C"].shape == (5, 2)
  assert parameters["W1"].shape == (6, 7)
  assert parameters["b1"].shape == (7,)
  assert parameters["W2"].shape == (7, 5)
  assert parameters["b2"].shape == (5,)
  assert all(parameter.is_leaf and parameter.requires_grad for parameter in parameters.values())


def test_logits_have_batch_by_vocabulary_shape_and_backpropagate():
  parameters = initialize_parameters(5, 3, 2, 7, seed=1)
  inputs = torch.tensor([[0, 1, 2], [2, 1, 0]])
  logits = mlp_logits(parameters, inputs)
  assert logits.shape == (2, 5)
  logits.sum().backward()
  assert all(parameter.grad is not None for parameter in parameters.values())


def test_evaluate_returns_cross_entropy_as_plain_float():
  parameters = initialize_parameters(3, 2, 2, 4, seed=1)
  inputs = torch.tensor([[0, 0], [0, 1], [1, 2]])
  targets = torch.tensor([1, 2, 0])
  expected = F.cross_entropy(mlp_logits(parameters, inputs), targets).item()
  result = evaluate_mlp_loss(parameters, inputs, targets)
  assert isinstance(result, float)
  assert result == pytest.approx(expected)


def test_train_reduces_full_dataset_loss():
  parameters = initialize_parameters(3, 2, 3, 12, seed=1)
  inputs = torch.tensor([[0, 0], [0, 1], [1, 2], [0, 0], [0, 1], [1, 2]])
  targets = torch.tensor([1, 2, 0, 1, 2, 0])
  before = evaluate_mlp_loss(parameters, inputs, targets)
  history = train_mlp(
    parameters,
    inputs,
    targets,
    steps=150,
    batch_size=6,
    learning_rate=0.1,
    seed=2,
  )
  after = evaluate_mlp_loss(parameters, inputs, targets)
  assert len(history) == 150
  assert after < before


def test_sample_respects_count_and_max_length():
  parameters = {
    "C": torch.zeros((2, 1), requires_grad=True),
    "W1": torch.zeros((2, 1), requires_grad=True),
    "b1": torch.zeros(1, requires_grad=True),
    "W2": torch.zeros((1, 2), requires_grad=True),
    "b2": torch.tensor([-100.0, 100.0], requires_grad=True),
  }
  samples = sample_mlp(
    parameters,
    {".": 0, "a": 1},
    {0: ".", 1: "a"},
    block_size=2,
    generator=torch.Generator().manual_seed(1),
    n_samples=3,
    max_length=4,
  )
  assert samples == ["aaaa"] * 3

