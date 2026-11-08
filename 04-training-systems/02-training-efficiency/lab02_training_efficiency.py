"""Lab 02: token-correct gradient accumulation before performance tuning."""

from collections.abc import Sequence

import torch


def microbatch_loss_weights(token_counts: Sequence[int]) -> list[float]:
  """Return n_i/sum(n) weights for losses that are per-microbatch token means."""
  # TODO 1: validate lengths and total count; preserve unequal microbatch sizes.
  raise NotImplementedError("TODO 1: implement microbatch_loss_weights")


def accumulate_microbatch_losses(
    losses: Sequence[torch.Tensor],
    token_counts: Sequence[int],
) -> None:
  """Backward scaled mean losses into existing gradients; do not step or zero."""
  # TODO 2: weight each graph by valid tokens and accumulate all contributions.
  raise NotImplementedError("TODO 2: implement accumulate_microbatch_losses")


def accumulation_step_boundaries(
    num_microbatches: int,
    accumulation_steps: int,
) -> list[int]:
  """Return zero-based microbatch indices after which optimizer.step must run."""
  # TODO 3: include every full accumulation group and a possible short tail.
  raise NotImplementedError("TODO 3: implement accumulation_step_boundaries")
