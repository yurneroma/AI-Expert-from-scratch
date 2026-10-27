"""Lab 02: fixed-context MLP character language model."""

from collections.abc import Sequence

import torch


Parameters = dict[str, torch.Tensor]


def split_words(
    words: Sequence[str],
    seed: int = 42,
    train_fraction: float = 0.8,
    dev_fraction: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
  """Shuffle reproducibly and split whole words into train, dev, and test."""
  # TODO 1: validate fractions, use a local seeded RNG, and avoid mutating input.
  raise NotImplementedError("TODO 1: implement split_words")


def build_context_dataset(
    words: Sequence[str],
    stoi: dict[str, int],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Create shape (N, block_size) contexts and shape (N,) targets."""
  # TODO 2: left-pad with boundary indices and shift after every target.
  raise NotImplementedError("TODO 2: implement build_context_dataset")


def initialize_parameters(
    vocab_size: int,
    block_size: int,
    embedding_dim: int,
    hidden_dim: int,
    seed: int = 2147483647,
) -> Parameters:
  """Create leaf tensors C, W1, b1, W2, and b2 with gradients enabled."""
  # TODO 3: use one seeded generator and dimensions derived from the arguments.
  raise NotImplementedError("TODO 3: implement initialize_parameters")


def mlp_logits(parameters: Parameters, inputs: torch.Tensor) -> torch.Tensor:
  """Run embedding lookup, context concatenation, tanh, and output affine."""
  # TODO 4a: keep the batch dimension for both B=1 and larger batches.
  raise NotImplementedError("TODO 4a: implement mlp_logits")


def evaluate_mlp_loss(
    parameters: Parameters,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
  """Evaluate mean cross entropy without constructing a retained graph."""
  # TODO 4b: call mlp_logits under no_grad and return a Python float.
  raise NotImplementedError("TODO 4b: implement evaluate_mlp_loss")


def train_mlp(
    parameters: Parameters,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 1_000,
    batch_size: int = 32,
    learning_rate: float = 0.1,
    seed: int = 42,
) -> list[float]:
  """Train in place with seeded mini-batch SGD and return scalar losses."""
  # TODO 5: sample batches, clear gradients, backpropagate, and update in no_grad.
  raise NotImplementedError("TODO 5: implement train_mlp")


def sample_mlp(
    parameters: Parameters,
    stoi: dict[str, int],
    itos: dict[int, str],
    block_size: int,
    generator: torch.Generator,
    n_samples: int = 10,
    max_length: int = 30,
) -> list[str]:
  """Generate fixed-context samples using the trained MLP."""
  # TODO 6: update context with every sampled token and stop at the boundary.
  raise NotImplementedError("TODO 6: implement sample_mlp")

