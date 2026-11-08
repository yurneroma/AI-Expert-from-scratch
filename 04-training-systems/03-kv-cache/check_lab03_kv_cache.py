"""Public cache lifecycle, capacity, memory, and full-prefix alignment checks."""

import math

import pytest
import torch

from lab03_kv_cache import KVCache, cached_attention_step


def test_cache_appends_memory_and_clear_resets_every_layer():
  cache = KVCache(num_layers=2, max_length=4)
  for layer in range(2):
    key = torch.randn(1, 1, 2, 3)
    value = torch.randn(1, 1, 2, 5)
    cache.update(layer, key, value)
  assert cache.length == 2
  initial_memory = cache.memory_bytes()
  assert initial_memory > 0
  cache.update(0, torch.randn(1, 1, 1, 3), torch.randn(1, 1, 1, 5))
  assert cache.length == 3 and cache.memory_bytes() > initial_memory
  cache.clear()
  assert cache.length == 0 and cache.memory_bytes() == 0


def test_cache_rejects_beyond_maximum_context():
  cache = KVCache(num_layers=1, max_length=1)
  with pytest.raises(ValueError, match="max"):
    cache.update(0, torch.randn(1, 1, 2, 3), torch.randn(1, 1, 2, 3))


def test_cached_logits_match_full_prefix_at_every_step():
  torch.manual_seed(11)
  q = torch.randn(2, 2, 4, 3, dtype=torch.float64)
  k = torch.randn(2, 2, 4, 3, dtype=torch.float64)
  v = torch.randn(2, 2, 4, 5, dtype=torch.float64)
  cache = KVCache(num_layers=1, max_length=4)
  for step in range(4):
    actual = cached_attention_step(
      q[:, :, step:step + 1], k[:, :, step:step + 1], v[:, :, step:step + 1], cache, 0
    )
    scores = q[:, :, step:step + 1] @ k[:, :, :step + 1].transpose(-2, -1) / math.sqrt(3)
    expected = scores.softmax(-1) @ v[:, :, :step + 1]
    assert torch.allclose(actual, expected)
    assert cache.length == step + 1
