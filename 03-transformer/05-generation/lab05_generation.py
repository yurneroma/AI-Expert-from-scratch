"""Lab 05: validated greedy/temperature/top-k autoregressive generation."""

from collections.abc import Callable, Sequence

import torch
from torch import nn


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
  """Sample class IDs from the final logits dimension with explicit controls."""
  # TODO 1: validate temperature/top-k, scale logits, filter, and sample IDs.
  raise NotImplementedError("TODO 1: implement sample_next_token")


def generate(
    model: nn.Module,
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[Sequence[int]], str],
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    context_length: int | None = None,
    bos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> str:
  """Return prompt+continuation; empty prompts require an explicit BOS ID."""
  # TODO 2: eval/no_grad, use last-position logits, honor EOS/max length/context,
  # sample with the supplied generator, and restore the original model mode.
  raise NotImplementedError("TODO 2: implement generate")
