"""Public staged checks for tokenizer and data Lab."""

import torch

from lab01_tokenizer import (
  ByteBPETokenizer,
  build_next_token_batch,
  byte_decode,
  byte_encode,
  split_documents,
)


def test_byte_round_trip_unicode_whitespace_and_empty():
  for text in ["hello", "中文", "a b\n", ""]:
    assert byte_decode(byte_encode(text)) == text


def test_document_split_is_reproducible_and_non_mutating():
  documents = [f"doc{i}" for i in range(10)]
  original = documents.copy()
  first = split_documents(documents, seed=7)
  assert first == split_documents(documents, seed=7)
  assert documents == original
  assert tuple(map(len, first)) == (8, 1, 1)


def test_bpe_learns_deterministic_merges_and_round_trips():
  tokenizer = ByteBPETokenizer.train(["abab"], num_merges=2)
  assert tokenizer.merges == [(97, 98), (256, 256)]
  assert tokenizer.encode("abab") == [257]
  assert tokenizer.decode(tokenizer.encode("中文 ab")) == "中文 ab"
  flagged = tokenizer.encode("ab", add_bos=True, add_eos=True)
  assert flagged[0] == tokenizer.bos_id and flagged[-1] == tokenizer.eos_id
  assert tokenizer.decode(flagged) == "ab"


def test_bpe_save_load_preserves_encoding(tmp_path):
  tokenizer = ByteBPETokenizer.train(["abab"], num_merges=2)
  path = tokenizer.save(tmp_path / "tokenizer.json")
  restored = ByteBPETokenizer.load(path)
  assert restored.merges == tokenizer.merges
  assert restored.encode("abab") == tokenizer.encode("abab")


def test_next_token_batch_has_exact_one_token_shift():
  X, Y = build_next_token_batch([0, 1, 2, 3, 4, 5], context_length=3, batch_size=2)
  assert X == [[0, 1, 2], [1, 2, 3]]
  assert Y == [[1, 2, 3], [2, 3, 4]]
  assert torch.tensor(Y).shape == (2, 3)
