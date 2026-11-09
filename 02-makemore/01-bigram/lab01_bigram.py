"""Lab 01: count-based and neural bigram language models.

Implement the TODOs in numerical order. Keep all returned tensors on CPU for
the public checks; device support can be added after correctness is established.
"""

import math

import torch


BOUNDARY = "."


def build_vocab(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
  """Build deterministic token maps with the boundary token at index zero.

  Contract:
    - ``stoi[BOUNDARY] == 0``
    - remaining characters are sorted
    - ``itos`` is the exact inverse of ``stoi``
  """
  chars = {char for word in words for char in word}
  chars.discard(BOUNDARY)
  tokens = [BOUNDARY, *sorted(chars)]
  stoi = {char: index for index, char in enumerate(tokens)}
  return stoi, dict(enumerate(tokens))


def build_bigram_counts(words: list[str], stoi: dict[str, int]) -> torch.Tensor:
  """Return an integer matrix where row=current and column=next token."""
  # 计数矩阵的大小为“词表大小 × 词表大小”，边界符同时表示单词的开头和结尾。
  vocab_size = len(stoi)
  boundary_index = stoi[BOUNDARY]

  # 给每个单词首尾补上边界符，再将相邻字符组成 bigram。
  # 例如 "ab" 会产生 .→a、a→b、b→. 三组计数。
  # 把二维索引 (current, next_token) 编码成一维整数，方便统一统计出现次数。
  pair_codes = [
    current * vocab_size + next_token
    for word_ids in ([stoi[char] for char in word] for word in words)
    for current, next_token in zip(
      (boundary_index, *word_ids),
      (*word_ids, boundary_index),
    )
  ]

  # 统计每种 bigram 的次数，再恢复成“行是当前字符、列是下一字符”的矩阵。
  # minlength 保证即使某些组合没有出现，输出仍包含完整的词表维度。
  return torch.bincount(
    torch.tensor(pair_codes, dtype=torch.int64),
    minlength=vocab_size * vocab_size,
  ).reshape(vocab_size, vocab_size)


def counts_to_probabilities(
    counts: torch.Tensor,
    smoothing: float = 1.0,
) -> torch.Tensor:
  """Normalize a square count table; require finite, nonnegative smoothing."""
  if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or counts.shape[0] == 0:
    raise ValueError("counts must be a non-empty square matrix")
  if not math.isfinite(smoothing) or smoothing < 0:
    raise ValueError("smoothing must be finite and nonnegative")

  smoothed_counts = counts.to(dtype=torch.get_default_dtype()) + smoothing
  row_totals = smoothed_counts.sum(dim=1, keepdim=True)
  safe_totals = torch.where(row_totals > 0, row_totals, torch.ones_like(row_totals))
  return torch.where(
    row_totals > 0,
    smoothed_counts / safe_totals,
    torch.full_like(smoothed_counts, 1.0 / counts.shape[1]),
  )


def average_nll(
    words: list[str],
    probabilities: torch.Tensor,
    stoi: dict[str, int],
) -> float:
  """Return mean NLL over boundary-aware bigrams; reject an empty word list."""
  if not words:
    raise ValueError("words must not be empty")

  vocab_size = probabilities.shape[1]
  boundary_index = stoi[BOUNDARY]
  pair_codes = [
    current * vocab_size + next_token
    for word_ids in ([stoi[char] for char in word] for word in words)
    for current, next_token in zip(
      (boundary_index, *word_ids),
      (*word_ids, boundary_index),
    )
  ]
  pair_indices = torch.tensor(pair_codes, dtype=torch.int64, device=probabilities.device)
  pair_probabilities = probabilities.reshape(-1)[pair_indices]
  return -pair_probabilities.log().mean().item()


def sample_bigram(
    probabilities: torch.Tensor,
    stoi: dict[str, int],
    itos: dict[int, str],
    generator: torch.Generator,
    n_samples: int = 10,
    max_length: int = 30,
) -> list[str]:
  """Generate words autoregressively from a bigram probability table."""
  boundary_index = stoi[BOUNDARY]
  words = []

  for _ in range(n_samples):
    current_index = boundary_index
    chars = []

    while len(chars) < max_length:
      next_index = int(
        torch.multinomial(
          probabilities[current_index],
          num_samples=1,
          generator=generator,
        ).item(),
      )
      if next_index == boundary_index:
        break

      chars.append(itos[next_index])
      current_index = next_index

    words.append("".join(chars))

  return words


def build_bigram_dataset(
    words: list[str],
    stoi: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return current-token inputs and next-token targets as int64 tensors."""
  boundary_index = stoi[BOUNDARY]
  inputs: list[int] = []
  targets: list[int] = []

  for word in words:
    word_ids = [stoi[char] for char in word]
    inputs.extend((boundary_index, *word_ids))
    targets.extend((*word_ids, boundary_index))

  return (
    torch.tensor(inputs, dtype=torch.int64),
    torch.tensor(targets, dtype=torch.int64),
  )


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

