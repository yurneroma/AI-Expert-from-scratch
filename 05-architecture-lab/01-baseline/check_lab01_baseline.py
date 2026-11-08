"""Public checks for frozen baseline manifests and controlled reuse."""

import pytest

from lab01_baseline import comparison_mismatches, validate_baseline_manifest


def manifest():
  return {
    "commit": "abc123",
    "tokenizer": {"type": "byte", "version": 1},
    "data_split": {"seed": 42, "train": "train.bin"},
    "model_config": {"layers": 2, "d_model": 64},
    "training_tokens": 100_000,
    "evaluation": {"batch_size": 32, "split": "dev"},
    "seed": 7,
  }


def test_complete_manifest_is_accepted_and_missing_fields_fail():
  assert validate_baseline_manifest(manifest())["training_tokens"] == 100_000
  incomplete = manifest()
  del incomplete["tokenizer"]
  with pytest.raises(ValueError, match="tokenizer"):
    validate_baseline_manifest(incomplete)


def test_comparison_reports_every_uncontrolled_shared_field():
  candidate = manifest()
  candidate["training_tokens"] = 200_000
  candidate["evaluation"] = {"batch_size": 16, "split": "dev"}
  assert comparison_mismatches(manifest(), candidate) == ["training_tokens", "evaluation"]
  assert comparison_mismatches(manifest(), manifest()) == []
