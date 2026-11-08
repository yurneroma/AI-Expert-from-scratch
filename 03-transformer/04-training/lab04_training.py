"""Lab 04: AdamW, schedules, and token-weighted evaluation."""

import math
from collections.abc import Iterable, Sequence

import torch
from torch import nn


def learning_rate_at_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float = 0.0,
) -> float:
  """Linear warmup followed by cosine decay for 0 <= step < total_steps."""
  # TODO 1: validate ranges and define warmup/decay boundaries explicitly.
  raise NotImplementedError("TODO 1: implement learning_rate_at_step")


class ManualAdamW:
  """Small AdamW implementation for dense leaf parameters and public checks."""

  def __init__(
      self,
      parameters: Iterable[nn.Parameter],
      lr: float = 1e-3,
      betas: tuple[float, float] = (0.9, 0.999),
      eps: float = 1e-8,
      weight_decay: float = 0.0,
  ) -> None:
    self.parameters = list(parameters)
    self.lr = lr
    self.betas = betas
    self.eps = eps
    self.weight_decay = weight_decay
    self.step_count = 0
    self.state = [
      {"m": torch.zeros_like(parameter), "v": torch.zeros_like(parameter)}
      for parameter in self.parameters
    ]

  def zero_grad(self) -> None:
    for parameter in self.parameters:
      parameter.grad = None

  @torch.no_grad()
  def step(self) -> None:
    """Update parameters in place using decoupled weight decay."""
    # TODO 2: use existing .grad, bias-correct m/v, and increment step once.
    raise NotImplementedError("TODO 2: implement ManualAdamW.step")


def weighted_token_nll(
    batch_loss_sums: Sequence[float],
    token_counts: Sequence[int],
) -> float:
  """Combine per-batch summed NLL using total valid tokens as the denominator."""
  # TODO 3: validate equal lengths/nonnegative counts and avoid division by zero.
  raise NotImplementedError("TODO 3: implement weighted_token_nll")


def evaluate_token_nll(
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> float:
  """Evaluate (B,T,V) logits over (B,T) targets without changing train mode."""
  # TODO 4: eval/no_grad, sum token NLL/counts, restore mode, return float.
  raise NotImplementedError("TODO 4: implement evaluate_token_nll")
