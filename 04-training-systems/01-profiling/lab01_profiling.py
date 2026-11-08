"""Lab 01: explicit resource accounting and repeatable timing helpers."""

import time
from collections.abc import Callable
from typing import Any

import torch
from torch import nn


def estimate_training_memory(
    parameter_numel: int,
    activation_numel: int = 0,
    attention_score_numel: int = 0,
    dtype_bytes: int = 4,
    optimizer_state_multiplier: int = 2,
) -> dict[str, int]:
  """Estimate parameters, gradients, optimizer, saved activations, scores, total."""
  # TODO 1: validate inputs and return a component-wise byte ledger.
  raise NotImplementedError("TODO 1: implement estimate_training_memory")


def benchmark_call(
    function: Callable[[], Any],
    warmup: int = 3,
    repeats: int = 10,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, float | int]:
  """Return repeat count and mean/median/stdev milliseconds after warmup."""
  # TODO 2: synchronize around timed calls when supplied; use perf_counter.
  raise NotImplementedError("TODO 2: implement benchmark_call")


def profile_training_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, float]:
  """Time one cross-entropy forward/backward/optimizer step; targets are class IDs."""
  # TODO 3: zero grad, time each phase, return finite loss and phase timings.
  raise NotImplementedError("TODO 3: implement profile_training_step")
