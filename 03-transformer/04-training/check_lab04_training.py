"""Public optimizer, schedule, and token-weighted evaluation checks."""

import math

import pytest
import torch
from torch import nn

from lab04_training import (
  ManualAdamW,
  evaluate_token_nll,
  learning_rate_at_step,
  weighted_token_nll,
)


def test_learning_rate_warmup_and_cosine_boundaries():
  values = [learning_rate_at_step(i, 10, 2, 1.0) for i in range(10)]
  assert values[0] == pytest.approx(0.5)
  assert values[1] == pytest.approx(1.0)
  assert values[-1] == pytest.approx(0.0)
  assert all(0 <= value <= 1 for value in values)
  with pytest.raises(ValueError):
    learning_rate_at_step(10, 10, 2, 1.0)


def test_manual_adamw_matches_pytorch_over_multiple_steps():
  torch.manual_seed(5)
  actual = nn.Parameter(torch.randn(3, dtype=torch.float64))
  expected = nn.Parameter(actual.detach().clone())
  manual = ManualAdamW([actual], lr=0.01, weight_decay=0.1)
  reference = torch.optim.AdamW([expected], lr=0.01, weight_decay=0.1)
  for step in range(3):
    gradient = torch.tensor([0.1 * (step + 1), -0.2, 0.3], dtype=torch.float64)
    actual.grad = gradient.clone()
    expected.grad = gradient.clone()
    manual.step()
    reference.step()
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
  assert manual.step_count == 3


def test_weighted_token_nll_uses_token_denominator():
  assert weighted_token_nll([2.0, 6.0], [2, 4]) == pytest.approx(4 / 3)
  with pytest.raises(ValueError):
    weighted_token_nll([1.0], [0])


def test_evaluation_is_batch_invariant_and_restores_training_mode():
  class TableModel(nn.Module):
    def __init__(self):
      super().__init__()
      self.weight = nn.Parameter(torch.arange(12, dtype=torch.float64).reshape(4, 3) / 10)

    def forward(self, inputs):
      return self.weight[inputs]

  model = TableModel().train()
  X = torch.tensor([[0, 1], [2, 3]])
  Y = torch.tensor([[1, 2], [3, 0]])
  before = model.weight.detach().clone()
  together = evaluate_token_nll(model, [(X, Y)])
  separate = evaluate_token_nll(model, [(X[:1], Y[:1]), (X[1:], Y[1:])])
  assert together == pytest.approx(separate)
  assert math.isfinite(together)
  assert model.training
  assert torch.equal(model.weight, before)
