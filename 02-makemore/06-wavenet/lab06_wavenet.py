"""Lab 06: a hierarchical, WaveNet-inspired context model."""

import sys
from pathlib import Path

import torch
from torch import nn

# This elective builds on the student's BatchNorm implementation from Lab 04.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04-batchnorm"))
from lab04_batchnorm import BatchNorm1d


class FlattenConsecutive(nn.Module):
  """Concatenate channels from adjacent groups along the time dimension."""

  def __init__(self, group_size: int = 2) -> None:
    super().__init__()
    if group_size < 2:
      raise ValueError("group_size must be at least two")
    self.group_size = group_size

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Map (B, T, C) to (B, T/group_size, C*group_size)."""
    # TODO 1: validate rank/divisibility and never remove the batch dimension.
    raise NotImplementedError("TODO 1: implement FlattenConsecutive.forward")


class HierarchicalBlock(nn.Module):
  """Group adjacent positions, project channels, normalize, and apply tanh."""

  def __init__(
      self,
      input_channels: int,
      output_channels: int,
      group_size: int = 2,
  ) -> None:
    super().__init__()
    # TODO 2a: register FlattenConsecutive, bias-free Linear, BatchNorm, and Tanh.
    raise NotImplementedError("TODO 2a: implement HierarchicalBlock.__init__")

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Return a sequence with a shorter time axis and output_channels features."""
    # TODO 2b: compose the registered operations in the documented order.
    raise NotImplementedError("TODO 2b: implement HierarchicalBlock.forward")


class WaveNetNameModel(nn.Module):
  """Repeated hierarchical blocks followed by a next-token classifier."""

  def __init__(
      self,
      vocab_size: int,
      block_size: int = 8,
      embedding_dim: int = 10,
      hidden_dim: int = 64,
      group_size: int = 2,
  ) -> None:
    super().__init__()
    # TODO 3a: store self.block_size, validate the hierarchy, and register
    # self.embedding, self.blocks, and self.head.
    raise NotImplementedError("TODO 3a: implement WaveNetNameModel.__init__")

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Return logits of shape (batch, vocab_size)."""
    # TODO 3b: run all levels and remove only the final singleton time axis.
    raise NotImplementedError("TODO 3b: implement WaveNetNameModel.forward")


def sample_wavenet(
    model: WaveNetNameModel,
    stoi: dict[str, int],
    itos: dict[int, str],
    generator: torch.Generator,
    n_samples: int = 10,
    max_length: int = 30,
) -> list[str]:
  """Generate samples using model.block_size tokens of left context."""
  # TODO 4: switch temporarily to eval, sample under no_grad, then restore mode.
  raise NotImplementedError("TODO 4: implement sample_wavenet")
