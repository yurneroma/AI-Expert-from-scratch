"""Lab 02: explicit, reference-first causal multi-head self-attention."""

import torch
from torch import nn


def split_heads(inputs: torch.Tensor, num_heads: int) -> torch.Tensor:
  """Map (B,T,D) to (B,H,T,D/H) without merging batch or time."""
  # TODO 1: validate divisibility and use reshape/transpose with clear axes.
  raise NotImplementedError("TODO 1: implement split_heads")


def merge_heads(inputs: torch.Tensor) -> torch.Tensor:
  """Map (B,H,T,d_head) back to contiguous (B,T,H*d_head)."""
  # TODO 1: restore token-major layout before reshaping.
  raise NotImplementedError("TODO 1: implement merge_heads")


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """Attention over (...,T,d); boolean mask entries are True when allowed."""
  # TODO 2: scale scores, apply mask before stable softmax, then multiply V.
  raise NotImplementedError("TODO 2: implement scaled_dot_product_attention")


class CausalSelfAttention(nn.Module):
  """Multi-head causal self-attention with packed QKV and output projection."""

  def __init__(self, d_model: int, num_heads: int) -> None:
    super().__init__()
    if d_model % num_heads != 0:
      raise ValueError("d_model must be divisible by num_heads")
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads
    self.qkv = nn.Linear(d_model, 3 * d_model)
    self.out_proj = nn.Linear(d_model, d_model)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Map (B,T,D) to (B,T,D), never attending to future positions."""
    # TODO 3: project/split QKV, build a causal mask, attend, merge and project.
    raise NotImplementedError("TODO 3: implement CausalSelfAttention.forward")
