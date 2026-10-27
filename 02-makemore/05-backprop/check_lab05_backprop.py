"""Public checks for Lab 05."""

import torch
import torch.nn.functional as F

from lab05_backprop import (
  compare_gradients,
  forward_loss,
  initialize_backprop_parameters,
  manual_backward,
)


def make_case():
  parameters = initialize_backprop_parameters(
    vocab_size=5,
    block_size=3,
    embedding_dim=2,
    hidden_dim=4,
    seed=7,
  )
  inputs = torch.tensor([
    [0, 0, 1],
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
  ])
  targets = torch.tensor([2, 3, 4, 0])
  return parameters, inputs, targets


def autograd_reference(parameters, inputs, targets):
  loss, cache = forward_loss(parameters, inputs, targets)
  for tensor in cache.values():
    if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
      tensor.retain_grad()
  loss.backward()
  return loss, cache


def test_forward_matches_fused_cross_entropy():
  parameters, inputs, targets = make_case()
  loss, cache = forward_loss(parameters, inputs, targets)
  assert torch.allclose(loss, F.cross_entropy(cache["logits"], targets))
  assert cache["hprebn"].shape == (4, 4)
  assert cache["logits"].shape == (4, 5)


def test_output_layer_manual_gradients_match_autograd():
  parameters, inputs, targets = make_case()
  _, cache = autograd_reference(parameters, inputs, targets)
  manual = manual_backward(parameters, cache)
  for name in ["logits", "h", "W2", "b2"]:
    reference = cache[name].grad if name in cache else parameters[name].grad
    assert name in manual
    assert torch.allclose(manual[name], reference, atol=1e-9, rtol=1e-7)


def test_batchnorm_manual_gradients_match_autograd():
  parameters, inputs, targets = make_case()
  _, cache = autograd_reference(parameters, inputs, targets)
  manual = manual_backward(parameters, cache)
  for name in ["hpreact", "bnraw", "hprebn", "bngain", "bnbias"]:
    reference = cache[name].grad if name in cache else parameters[name].grad
    assert name in manual
    assert torch.allclose(manual[name], reference, atol=1e-9, rtol=1e-7)


def test_all_parameters_match_autograd_and_report_is_complete():
  parameters, inputs, targets = make_case()
  _, cache = autograd_reference(parameters, inputs, targets)
  manual = manual_backward(parameters, cache)
  for name, parameter in parameters.items():
    assert name in manual
    assert torch.allclose(manual[name], parameter.grad, atol=1e-9, rtol=1e-7)
  report = compare_gradients(manual, parameters)
  assert set(report) == set(parameters)
  assert all(item["close"] for item in report.values())
  assert all(set(item) == {"close", "exact", "max_abs_error", "relative_error"}
             for item in report.values())

