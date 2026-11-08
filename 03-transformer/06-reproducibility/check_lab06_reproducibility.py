"""Public checkpoint, optimizer, RNG, and compatibility checks."""

import random

import pytest
import torch
from torch import nn

from lab06_reproducibility import (
  capture_rng_state,
  load_checkpoint,
  restore_rng_state,
  save_checkpoint,
)


def test_rng_restore_repeats_python_and_torch_sequences():
  random.seed(7)
  torch.manual_seed(7)
  state = capture_rng_state()
  expected = (random.random(), torch.randn(3))
  random.random()
  torch.randn(4)
  restore_rng_state(state)
  actual = (random.random(), torch.randn(3))
  assert actual[0] == expected[0]
  assert torch.equal(actual[1], expected[1])


def test_checkpoint_restores_logits_optimizer_and_training_position(tmp_path):
  torch.manual_seed(3)
  model = nn.Linear(3, 2)
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
  loss = model(torch.ones(1, 3)).sum()
  loss.backward()
  optimizer.step()
  config = {"d_model": 3, "vocab_size": 2, "context_length": 4}
  path = save_checkpoint(
    tmp_path / "checkpoint.pt", model, optimizer, 5, config,
    {"type": "byte", "merges": []}, data_position=128,
  )
  restored = nn.Linear(3, 2)
  restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
  state = load_checkpoint(path, restored, restored_optimizer, expected_config=config)
  inputs = torch.randn(2, 3)
  assert torch.allclose(restored(inputs), model(inputs))
  assert state["step"] == 5 and state["data_position"] == 128
  assert restored_optimizer.state


def test_checkpoint_rejects_incompatible_configuration(tmp_path):
  model = nn.Linear(2, 2)
  path = save_checkpoint(tmp_path / "model.pt", model, None, 0, {"width": 2}, {})
  with pytest.raises(ValueError, match="config"):
    load_checkpoint(path, nn.Linear(2, 2), expected_config={"width": 3})
