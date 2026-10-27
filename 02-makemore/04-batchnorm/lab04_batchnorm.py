"""Lab 04: BatchNorm1d and a deep MLP assembled from explicit blocks."""

import torch
from torch import nn


class BatchNorm1d(nn.Module):
  """Normalize the final feature dimension for 2-D or 3-D inputs."""

  def __init__(
      self,
      num_features: int,
      eps: float = 1e-5,
      momentum: float = 0.1,
  ) -> None:
    super().__init__()
    # TODO 1: create self.gamma/self.beta Parameters, self.running_mean/
    # self.running_var buffers, and store eps/momentum.
    raise NotImplementedError("TODO 1: implement BatchNorm1d.__init__")

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Use batch statistics in training and running statistics in eval."""
    # TODO 2: support (B, C) and (B, T, C); update buffers only in training.
    raise NotImplementedError("TODO 2: implement BatchNorm1d.forward")


class BatchNormMLP(nn.Module):
  """Embedding MLP with repeated Linear -> BatchNorm -> Tanh blocks."""

  def __init__(
      self,
      vocab_size: int,
      block_size: int,
      embedding_dim: int,
      hidden_dims: list[int],
  ) -> None:
    super().__init__()
    # TODO 3: create self.embedding, self.blocks, and self.head. Hidden Linear
    # modules should omit bias because BatchNorm supplies the learned offset.
    raise NotImplementedError("TODO 3: implement BatchNormMLP.__init__")

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Return logits of shape (batch, vocab_size)."""
    # TODO 4: preserve the batch dimension and compose registered modules.
    raise NotImplementedError("TODO 4: implement BatchNormMLP.forward")
