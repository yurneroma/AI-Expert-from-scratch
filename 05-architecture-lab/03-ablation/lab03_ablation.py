"""Lab 03: paired-seed ablation validation and reproducible summaries."""

import math
import statistics
from typing import Any


SHARED_FIELDS = (
  "tokenizer",
  "data_split",
  "training_tokens",
  "effective_batch",
  "optimizer",
  "lr_schedule",
)


def validate_paired_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Require two variants, at least three paired seeds, and shared controls."""
  # TODO 1: validate variant/seed/config/metrics and return the unchanged runs.
  raise NotImplementedError("TODO 1: implement validate_paired_runs")


def summarize_runs(
    runs: list[dict[str, Any]],
    metrics: tuple[str, ...] = ("dev_nll", "tokens_per_second"),
) -> dict[str, dict[str, dict[str, float]]]:
  """Return per-variant metric mean and sample standard deviation."""
  # TODO 2: validate runs, require finite metrics, and summarize every seed.
  raise NotImplementedError("TODO 2: implement summarize_runs")
