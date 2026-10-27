"""Lab 05: manually differentiate a small batch-normalized MLP."""

import torch


Parameters = dict[str, torch.Tensor]


def initialize_backprop_parameters(
    vocab_size: int,
    block_size: int,
    embedding_dim: int,
    hidden_dim: int,
    seed: int = 42,
) -> Parameters:
  """Create a reproducible parameter set used by the forward/backward lab."""
  generator = torch.Generator().manual_seed(seed)
  dtype = torch.float64
  parameters = {
    "C": torch.randn((vocab_size, embedding_dim), generator=generator, dtype=dtype),
    "W1": torch.randn(
      (block_size * embedding_dim, hidden_dim), generator=generator, dtype=dtype
    ) * ((5 / 3) / (block_size * embedding_dim) ** 0.5),
    "b1": torch.randn(hidden_dim, generator=generator, dtype=dtype) * 0.01,
    "bngain": torch.ones((1, hidden_dim), dtype=dtype),
    "bnbias": torch.zeros((1, hidden_dim), dtype=dtype),
    "W2": torch.randn(
      (hidden_dim, vocab_size), generator=generator, dtype=dtype
    ) * 0.1,
    "b2": torch.randn(vocab_size, generator=generator, dtype=dtype) * 0.1,
  }
  for parameter in parameters.values():
    parameter.requires_grad_()
  return parameters


def forward_loss(
    parameters: Parameters,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  """Run an explicit primitive-level forward pass and return its cache."""
  C = parameters["C"]
  W1, b1 = parameters["W1"], parameters["b1"]
  bngain, bnbias = parameters["bngain"], parameters["bnbias"]
  W2, b2 = parameters["W2"], parameters["b2"]

  emb = C[inputs]
  embcat = emb.reshape(emb.shape[0], -1)
  hprebn = embcat @ W1 + b1
  bnmean = hprebn.mean(dim=0, keepdim=True)
  bndiff = hprebn - bnmean
  bndiff2 = bndiff**2
  bnvar = bndiff2.mean(dim=0, keepdim=True)
  bnvar_inv = (bnvar + eps) ** -0.5
  bnraw = bndiff * bnvar_inv
  hpreact = bngain * bnraw + bnbias
  h = torch.tanh(hpreact)
  logits = h @ W2 + b2
  log_probs = torch.log_softmax(logits, dim=1)
  loss = -log_probs[torch.arange(inputs.shape[0]), targets].mean()

  cache = {
    "inputs": inputs,
    "targets": targets,
    "emb": emb,
    "embcat": embcat,
    "hprebn": hprebn,
    "bnmean": bnmean,
    "bndiff": bndiff,
    "bndiff2": bndiff2,
    "bnvar": bnvar,
    "bnvar_inv": bnvar_inv,
    "bnraw": bnraw,
    "hpreact": hpreact,
    "h": h,
    "logits": logits,
    "log_probs": log_probs,
    "eps": torch.tensor(eps, dtype=hprebn.dtype),
  }
  return loss, cache


def manual_backward(
    parameters: Parameters,
    cache: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
  """Return gradients for intermediates and every parameter.

  Required parameter keys are ``C``, ``W1``, ``b1``, ``bngain``, ``bnbias``,
  ``W2``, and ``b2``. Public milestones also inspect ``logits``, ``h``,
  ``hpreact``, ``bnraw``, and ``hprebn`` gradients.
  """
  # TODO 1: reverse forward_loss without invoking Tensor.backward or autograd.grad.
  raise NotImplementedError("TODO 1: implement manual_backward")


def compare_gradients(
    manual: dict[str, torch.Tensor],
    parameters: Parameters,
    atol: float = 1e-9,
    rtol: float = 1e-7,
) -> dict[str, dict[str, float | bool]]:
  """Compare manual parameter gradients with populated autograd gradients."""
  # TODO 2: report close, exact, max_abs_error, and relative_error per parameter.
  raise NotImplementedError("TODO 2: implement compare_gradients")
