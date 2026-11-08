"""Lab 01: byte fallback, simplified byte-BPE, and next-token batches."""

import json
from collections.abc import Sequence
from pathlib import Path


BOS_TEXT = "<|bos|>"
EOS_TEXT = "<|eos|>"


def byte_encode(text: str) -> list[int]:
  """Encode text as UTF-8 byte IDs in the range 0..255."""
  # TODO 1: use UTF-8 bytes without adding implicit special tokens.
  raise NotImplementedError("TODO 1: implement byte_encode")


def byte_decode(token_ids: Sequence[int]) -> str:
  """Decode raw UTF-8 byte IDs; reject merged or special IDs clearly."""
  # TODO 1: validate IDs and round-trip arbitrary Unicode text.
  raise NotImplementedError("TODO 1: implement byte_decode")


def split_documents(
    documents: Sequence[str],
    seed: int = 42,
    train_fraction: float = 0.8,
    dev_fraction: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
  """Split whole documents reproducibly without mutating the input."""
  # TODO 2: validate fractions and use a local seeded RNG.
  raise NotImplementedError("TODO 2: implement split_documents")


class ByteBPETokenizer:
  """Small byte-BPE with IDs 0..255 and deterministic merge ranks."""

  def __init__(self, merges: Sequence[tuple[int, int]] = ()) -> None:
    self.merges = list(merges)
    self.bos_id = 256 + len(self.merges)
    self.eos_id = self.bos_id + 1

  @classmethod
  def train(cls, documents: Sequence[str], num_merges: int) -> "ByteBPETokenizer":
    """Learn at most num_merges from training documents only."""
    # TODO 3: count adjacent pairs, break count ties by pair order, and merge.
    raise NotImplementedError("TODO 3: implement ByteBPETokenizer.train")

  def encode(
      self,
      text: str,
      add_bos: bool = False,
      add_eos: bool = False,
  ) -> list[int]:
    """Apply merges in learned order and optionally add BOS/EOS."""
    # TODO 4: start from UTF-8 bytes and apply each merge in rank order.
    raise NotImplementedError("TODO 4: implement ByteBPETokenizer.encode")

  def decode(self, token_ids: Sequence[int]) -> str:
    """Recursively expand merged IDs and decode UTF-8; ignore BOS/EOS."""
    # TODO 4: validate IDs and preserve empty/text round-trips.
    raise NotImplementedError("TODO 4: implement ByteBPETokenizer.decode")

  def save(self, path: str | Path) -> Path:
    """Save the ordered merge table as portable JSON."""
    # TODO 5: write UTF-8 JSON and return the path.
    raise NotImplementedError("TODO 5: implement ByteBPETokenizer.save")

  @classmethod
  def load(cls, path: str | Path) -> "ByteBPETokenizer":
    """Load the exact merge order saved by save()."""
    # TODO 5: validate the JSON structure and reconstruct special IDs.
    raise NotImplementedError("TODO 5: implement ByteBPETokenizer.load")


def build_next_token_batch(
    tokens: Sequence[int],
    context_length: int,
    batch_size: int,
    start: int = 0,
) -> tuple[list[list[int]], list[list[int]]]:
  """Build consecutive windows with targets shifted by exactly one token."""
  # TODO 6: return X/Y lists and reject insufficient tokens or invalid ranges.
  raise NotImplementedError("TODO 6: implement build_next_token_batch")
