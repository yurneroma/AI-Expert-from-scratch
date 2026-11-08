"""Lab 01: machine-checkable baseline and comparison contracts."""

from typing import Any


REQUIRED_FIELDS = {
  "commit",
  "tokenizer",
  "data_split",
  "model_config",
  "training_tokens",
  "evaluation",
  "seed",
}
SHARED_COMPARISON_FIELDS = ("tokenizer", "data_split", "training_tokens", "evaluation")


def validate_baseline_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
  """Validate and return a complete frozen-baseline manifest."""
  # TODO 1: require all fields, positive token budget, and a model config mapping.
  raise NotImplementedError("TODO 1: implement validate_baseline_manifest")


def comparison_mismatches(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
  """List shared fields that differ and would make a comparison uncontrolled."""
  # TODO 2: validate both manifests and return deterministic differing field names.
  raise NotImplementedError("TODO 2: implement comparison_mismatches")
