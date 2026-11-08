"""Public correctness checks for unequal-token accumulation and tail steps."""

import pytest
import torch

from lab02_training_efficiency import (
  accumulate_microbatch_losses,
  accumulation_step_boundaries,
  microbatch_loss_weights,
)


def test_loss_weights_respect_unequal_token_counts():
  assert microbatch_loss_weights([2, 6]) == pytest.approx([0.25, 0.75])
  with pytest.raises(ValueError):
    microbatch_loss_weights([0, 0])


def test_accumulated_gradient_matches_effective_large_batch():
  actual = torch.tensor([1.0, -2.0], requires_grad=True)
  expected = actual.detach().clone().requires_grad_()
  losses = [(actual * torch.tensor([1.0, 2.0])).mean(), (actual**2).sum()]
  counts = [2, 1]
  accumulate_microbatch_losses(losses, counts)
  reference_losses = [
    (expected * torch.tensor([1.0, 2.0])).mean(),
    (expected**2).sum(),
  ]
  reference_loss = (reference_losses[0] * 2 + reference_losses[1]) / 3
  reference_loss.backward()
  assert torch.allclose(actual.grad, expected.grad)


def test_accumulation_boundaries_include_short_tail():
  assert accumulation_step_boundaries(6, 2) == [1, 3, 5]
  assert accumulation_step_boundaries(5, 2) == [1, 3, 4]
  with pytest.raises(ValueError):
    accumulation_step_boundaries(0, 2)
