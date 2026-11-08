"""Lab 07: validate comparison and controlled-ablation result tables."""

import math
from typing import Any


REQUIRED_MODELS = {"Count bigram", "Neural bigram", "MLP"}
NUMERIC_FIELDS = ("parameters", "train_nll", "dev_nll", "test_nll", "training_time")


def validate_comparison_rows(rows: list[dict[str, Any]]) -> None:
  """Require complete finite results for the three mandatory models."""
  # TODO 1: require unique model names, all mandatory models, finite numerics.
  raise NotImplementedError("TODO 1: implement validate_comparison_rows")


def validate_ablation_rows(rows: list[dict[str, Any]]) -> None:
  """Require at least three distinct settings with one controlled variable change."""
  # TODO 2: require shared controls, one independent variable, finite results.
  raise NotImplementedError("TODO 2: implement validate_ablation_rows")
