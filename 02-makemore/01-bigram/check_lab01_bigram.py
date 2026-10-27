"""Public checks for Lab 01, ordered by the handout milestones."""

import math

import pytest
import torch

from lab01_bigram import (
  average_nll,
  build_bigram_counts,
  build_bigram_dataset,
  build_vocab,
  counts_to_probabilities,
  neural_bigram_loss,
  sample_bigram,
  train_neural_bigram,
)


@pytest.fixture
def tiny_vocab():
  return build_vocab(["ab", "aa"])


def test_vocab_is_deterministic_and_invertible(tiny_vocab):
  stoi, itos = tiny_vocab
  assert stoi == {".": 0, "a": 1, "b": 2}
  assert itos == {0: ".", 1: "a", 2: "b"}


def test_counts_include_start_end_and_repeated_pairs(tiny_vocab):
  stoi, _ = tiny_vocab
  counts = build_bigram_counts(["ab", "aa"], stoi)
  expected = torch.tensor([
    [0, 2, 0],
    [1, 1, 1],
    [1, 0, 0],
  ])
  assert counts.dtype == torch.int64
  assert torch.equal(counts, expected)


def test_probabilities_apply_smoothing_per_row():
  counts = torch.tensor([[1, 1], [0, 2]], dtype=torch.long)
  probabilities = counts_to_probabilities(counts, smoothing=1.0)
  expected = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
  assert probabilities.is_floating_point()
  assert torch.allclose(probabilities, expected)
  assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))


def test_nll_matches_uniform_distribution(tiny_vocab):
  stoi, _ = tiny_vocab
  probabilities = torch.full((3, 3), 1 / 3)
  result = average_nll(["ab", "aa"], probabilities, stoi)
  assert result == pytest.approx(math.log(3), abs=1e-7)


def test_sample_starts_and_stops_at_boundary():
  stoi = {".": 0, "a": 1}
  itos = {0: ".", 1: "a"}
  probabilities = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
  samples = sample_bigram(
    probabilities,
    stoi,
    itos,
    torch.Generator().manual_seed(7),
    n_samples=4,
  )
  assert samples == ["a"] * 4


def test_dataset_contains_one_example_per_transition(tiny_vocab):
  stoi, _ = tiny_vocab
  inputs, targets = build_bigram_dataset(["ab"], stoi)
  assert inputs.dtype == targets.dtype == torch.int64
  assert torch.equal(inputs, torch.tensor([0, 1, 2]))
  assert torch.equal(targets, torch.tensor([1, 2, 0]))


def test_neural_loss_is_differentiable_and_matches_uniform(tiny_vocab):
  stoi, _ = tiny_vocab
  inputs, targets = build_bigram_dataset(["ab", "aa"], stoi)
  weights = torch.zeros((3, 3), requires_grad=True)
  result = neural_bigram_loss(weights, inputs, targets)
  assert result.item() == pytest.approx(math.log(3), abs=1e-7)
  result.backward()
  assert weights.grad is not None
  assert torch.isfinite(weights.grad).all()


def test_neural_training_reduces_loss_and_returns_history(tiny_vocab):
  stoi, _ = tiny_vocab
  inputs, targets = build_bigram_dataset(["ab", "aa"] * 10, stoi)
  weights, history = train_neural_bigram(
    inputs,
    targets,
    vocab_size=3,
    steps=80,
    learning_rate=5.0,
    regularization=0.0,
    seed=7,
  )
  assert weights.shape == (3, 3)
  assert not weights.requires_grad
  assert len(history) == 80
  assert history[-1] < history[0]

