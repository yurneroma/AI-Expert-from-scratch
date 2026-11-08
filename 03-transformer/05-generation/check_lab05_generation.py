"""Public sampling, EOS, determinism, context, and state checks."""

import pytest
import torch
from torch import nn

from lab05_generation import generate, sample_next_token


def test_top_k_one_is_greedy_and_seed_is_reproducible():
  logits = torch.tensor([0.0, 1.0, 3.0, 2.0])
  assert sample_next_token(logits, top_k=1).item() == 2
  first = sample_next_token(logits, generator=torch.Generator().manual_seed(8))
  second = sample_next_token(logits, generator=torch.Generator().manual_seed(8))
  assert first == second
  with pytest.raises(ValueError):
    sample_next_token(logits, temperature=0)
  with pytest.raises(ValueError):
    sample_next_token(logits, top_k=9)


class FixedModel(nn.Module):
  def __init__(self, next_id=2):
    super().__init__()
    self.anchor = nn.Parameter(torch.tensor(0.0))
    self.next_id = next_id
    self.lengths = []

  def forward(self, inputs):
    self.lengths.append(inputs.shape[1])
    logits = torch.full((*inputs.shape, 4), -10.0, device=inputs.device)
    logits[..., self.next_id] = 10.0
    return logits + self.anchor


def test_generate_eos_context_and_restores_training_mode():
  model = FixedModel(next_id=2).train()
  before = model.anchor.detach().clone()
  text = generate(
    model,
    encode=lambda text: [int(c) for c in text],
    decode=lambda ids: "".join(str(i) for i in ids),
    prompt="0",
    max_new_tokens=5,
    top_k=1,
    eos_id=2,
    context_length=2,
  )
  assert text == "02"
  assert model.training
  assert torch.equal(model.anchor, before)


def test_generate_empty_prompt_uses_bos_and_max_length():
  model = FixedModel(next_id=1).eval()
  text = generate(
    model,
    encode=str.encode,
    decode=lambda ids: "".join(map(str, ids)),
    prompt="",
    bos_id=0,
    max_new_tokens=3,
    top_k=1,
    context_length=2,
  )
  assert text == "0111"
  assert not model.training
  assert max(model.lengths) <= 2
