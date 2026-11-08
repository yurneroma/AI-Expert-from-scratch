"""Lab 02: one switchable component change, defaulting to RMSNorm."""

import torch
from torch import nn


class RMSNorm(nn.Module):
  """RMS-normalize the final feature dimension with a learned gain."""

  def __init__(self, d_model: int, eps: float = 1e-5) -> None:
    super().__init__()
    if d_model <= 0 or eps <= 0:
      raise ValueError("d_model and eps must be positive")
    self.d_model = d_model
    self.eps = eps
    self.gain = nn.Parameter(torch.ones(d_model))

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Preserve input shape and normalize only the final feature dimension."""
    # TODO 1: mean square over dim=-1, stable rsqrt, then learned gain.
    raise NotImplementedError("TODO 1: implement RMSNorm.forward")


def build_norm(kind: str, d_model: int, eps: float = 1e-5) -> nn.Module:
  """Build the unchanged LayerNorm baseline or RMSNorm variant."""
  # TODO 2: support 'layer_norm' and 'rmsnorm'; reject unknown components.
  raise NotImplementedError("TODO 2: implement build_norm")
