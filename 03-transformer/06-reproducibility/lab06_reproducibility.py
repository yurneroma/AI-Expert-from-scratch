"""Lab 06: explicit checkpoint contracts and RNG restoration."""

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn


def capture_rng_state() -> dict[str, Any]:
  """Capture Python and torch CPU RNG state used by the course scripts."""
  # TODO 1: return independent snapshots, not mutable references.
  raise NotImplementedError("TODO 1: implement capture_rng_state")


def restore_rng_state(state: dict[str, Any]) -> None:
  """Restore a state produced by capture_rng_state."""
  # TODO 1: validate keys and restore both RNGs.
  raise NotImplementedError("TODO 1: implement restore_rng_state")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: dict[str, Any],
    tokenizer_config: dict[str, Any],
    scheduler: Any | None = None,
    data_position: int = 0,
) -> Path:
  """Save model/training state plus structure, tokenizer, RNG, and data position."""
  # TODO 2: include enough state to resume, create parent directory, return path.
  raise NotImplementedError("TODO 2: implement save_checkpoint")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    expected_config: dict[str, Any] | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
  """Load with explicit architecture compatibility and return metadata/state."""
  # TODO 3: validate config before loading, restore requested states/RNG.
  raise NotImplementedError("TODO 3: implement load_checkpoint")
