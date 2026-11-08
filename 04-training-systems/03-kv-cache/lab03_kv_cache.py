"""Lab 03: a reference KV cache with explicit layout and lifecycle."""

import math

import torch


class KVCache:
  """Per-layer K/V tensors in (B,H,T,d_head) layout."""

  def __init__(self, num_layers: int, max_length: int) -> None:
    if num_layers <= 0 or max_length <= 0:
      raise ValueError("num_layers and max_length must be positive")
    self.num_layers = num_layers
    self.max_length = max_length
    self.keys: list[torch.Tensor | None] = [None] * num_layers
    self.values: list[torch.Tensor | None] = [None] * num_layers

  @property
  def length(self) -> int:
    """Current cached sequence length, or zero for an empty cache."""
    return 0 if self.keys[0] is None else self.keys[0].shape[2]

  def update(
      self,
      layer_index: int,
      key: torch.Tensor,
      value: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one layer's K/V along time and return all available K/V."""
    # TODO 1: validate layer/rank/shape/dtype/device, append on time, enforce max.
    raise NotImplementedError("TODO 1: implement KVCache.update")

  def clear(self) -> None:
    """Remove all layer state so independent generations cannot leak cache."""
    # TODO 2: reset every layer while retaining the configuration.
    raise NotImplementedError("TODO 2: implement KVCache.clear")

  def memory_bytes(self) -> int:
    """Return bytes held by all cached K/V tensors."""
    # TODO 3: account for both K and V using each tensor's actual element size.
    raise NotImplementedError("TODO 3: implement KVCache.memory_bytes")


def cached_attention_step(
    query: torch.Tensor,
    key_new: torch.Tensor,
    value_new: torch.Tensor,
    cache: KVCache,
    layer_index: int,
) -> torch.Tensor:
  """Append new K/V and attend from current queries to all cached positions."""
  # TODO 4: update cache, use sqrt(d_head), softmax over cached time, multiply V.
  raise NotImplementedError("TODO 4: implement cached_attention_step")
