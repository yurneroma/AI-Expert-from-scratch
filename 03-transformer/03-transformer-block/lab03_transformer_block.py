"""Lab 03: Pre-LN decoder blocks and a small complete GPT."""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02-attention"))
from lab02_attention import CausalSelfAttention  # noqa: E402


@dataclass(frozen=True)
class TransformerConfig:
  vocab_size: int
  context_length: int
  d_model: int
  num_heads: int
  num_layers: int
  ffn_dim: int

  def __post_init__(self) -> None:
    if min(self.vocab_size, self.context_length, self.d_model, self.num_layers) <= 0:
      raise ValueError("model dimensions must be positive")
    if self.d_model % self.num_heads != 0:
      raise ValueError("d_model must be divisible by num_heads")
    if self.ffn_dim <= 0:
      raise ValueError("ffn_dim must be positive")


class TransformerBlock(nn.Module):
  """Pre-LN attention and FFN, each followed by a residual addition."""

  def __init__(self, config: TransformerConfig) -> None:
    super().__init__()
    self.norm1 = nn.LayerNorm(config.d_model)
    self.attention = CausalSelfAttention(config.d_model, config.num_heads)
    self.norm2 = nn.LayerNorm(config.d_model)
    self.ffn = nn.Sequential(
      nn.Linear(config.d_model, config.ffn_dim),
      nn.GELU(),
      nn.Linear(config.ffn_dim, config.d_model),
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Preserve (B,T,D) and the causal constraint."""
    # TODO 1: compose two Pre-LN sublayers and two residual additions.
    raise NotImplementedError("TODO 1: implement TransformerBlock.forward")


class TinyGPT(nn.Module):
  """Token/position embeddings, decoder blocks, final norm, and vocab head."""

  def __init__(self, config: TransformerConfig) -> None:
    super().__init__()
    self.config = config
    self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
    self.position_embedding = nn.Embedding(config.context_length, config.d_model)
    self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
    self.final_norm = nn.LayerNorm(config.d_model)
    self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Map integer (B,T) tokens to (B,T,vocab_size) logits."""
    # TODO 2: validate rank/length, add positions, run blocks/norm/head.
    raise NotImplementedError("TODO 2: implement TinyGPT.forward")


def count_parameters(model: nn.Module) -> int:
  """Return the number of scalar parameters without counting buffers."""
  return sum(parameter.numel() for parameter in model.parameters())
