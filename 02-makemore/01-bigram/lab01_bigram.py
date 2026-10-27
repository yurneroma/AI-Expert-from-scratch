"""Lab 01: count-based and neural bigram language models.

Implement the TODOs in numerical order. Keep all returned tensors on CPU for
the public checks; device support can be added after correctness is established.
"""

import torch


BOUNDARY = "."


def build_vocab(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
  """Build deterministic token maps with the boundary token at index zero.

  Contract:
    - ``stoi[BOUNDARY] == 0``
    - remaining characters are sorted
    - ``itos`` is the exact inverse of ``stoi``
  """
  # TODO 1: collect, sort, and index the vocabulary.
  raise NotImplementedError("TODO 1: implement build_vocab")


def build_bigram_counts(words: list[str], stoi: dict[str, int]) -> torch.Tensor:
  """Return an integer matrix where row=current and column=next token."""
  # TODO 2: include both start and end boundary transitions for every word.
  raise NotImplementedError("TODO 2: implement build_bigram_counts")


def counts_to_probabilities(
    counts: torch.Tensor,
    smoothing: float = 1.0,
) -> torch.Tensor:
  """Apply additive smoothing and normalize each row to sum to one."""
  # TODO 3: preserve row semantics and return floating-point probabilities.
  raise NotImplementedError("TODO 3: implement counts_to_probabilities")


def average_nll(
    words: list[str],
    probabilities: torch.Tensor,
    stoi: dict[str, int],
) -> float:
  """Return mean negative log-likelihood over all boundary-aware bigrams."""
  # TODO 4: index P[current, next], accumulate log probabilities, and average.
  raise NotImplementedError("TODO 4: implement average_nll")


def sample_bigram(
    probabilities: torch.Tensor,
    stoi: dict[str, int],
    itos: dict[int, str],
    generator: torch.Generator,
    n_samples: int = 10,
    max_length: int = 30,
) -> list[str]:
  """Generate words autoregressively from a bigram probability table."""
  # TODO 5: begin at boundary, sample until boundary, and enforce max_length.
  raise NotImplementedError("TODO 5: implement sample_bigram")


def build_bigram_dataset(
    words: list[str],
    stoi: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return current-token inputs and next-token targets as int64 tensors."""
  # TODO 6: flatten all boundary-aware bigrams into two one-dimensional tensors.
  raise NotImplementedError("TODO 6: implement build_bigram_dataset")


def neural_bigram_loss(
    weights: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    regularization: float = 0.0,
) -> torch.Tensor:
  """Return cross-entropy plus optional mean-square weight regularization."""
  # TODO 7a: select one logits row per input and compute a differentiable loss.
  raise NotImplementedError("TODO 7a: implement neural_bigram_loss")


def train_neural_bigram(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    steps: int = 100,
    learning_rate: float = 20.0,
    regularization: float = 0.01,
    seed: int = 2147483647,
) -> tuple[torch.Tensor, list[float]]:
  """Train logits with full-batch gradient descent and return loss history."""
  # TODO 7b: initialize a leaf tensor, train it, and return detached weights.
  raise NotImplementedError("TODO 7b: implement train_neural_bigram")

