"""Public memory-accounting and timing-contract checks."""

import pytest
import torch
from torch import nn

from lab01_profiling import (
  benchmark_call,
  estimate_training_memory,
  profile_training_step,
)


def test_memory_estimate_separates_parameters_gradients_optimizer_and_activations():
  result = estimate_training_memory(100, activation_numel=40, attention_score_numel=20)
  assert result["parameters"] == 400
  assert result["gradients"] == 400
  assert result["optimizer_states"] == 800
  assert result["activations"] == 160
  assert result["attention_scores"] == 80
  assert result["total"] == 1840
  with pytest.raises(ValueError):
    estimate_training_memory(-1)


def test_benchmark_runs_warmup_and_reports_repeats():
  calls = []
  result = benchmark_call(lambda: calls.append(1), warmup=2, repeats=5)
  assert len(calls) == 7
  assert result["repeats"] == 5
  assert result["median_ms"] >= 0 and result["stdev_ms"] >= 0


def test_training_profile_covers_one_real_step():
  model = nn.Linear(3, 2)
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
  before = [parameter.detach().clone() for parameter in model.parameters()]
  result = profile_training_step(
    model, torch.randn(4, 3), torch.randint(0, 2, (4,)), optimizer
  )
  assert result["loss"] > 0
  assert result["forward_ms"] >= 0 and result["backward_ms"] >= 0
  assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
