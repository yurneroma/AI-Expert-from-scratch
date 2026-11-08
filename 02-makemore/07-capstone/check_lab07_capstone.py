"""Public checks for complete comparisons and controlled ablations."""

import pytest

from lab07_capstone import validate_ablation_rows, validate_comparison_rows


def comparison_rows():
  return [
    {
      "model": model,
      "parameters": 100,
      "train_nll": 1.0,
      "dev_nll": 1.2,
      "test_nll": 1.3,
      "training_time": 10.0,
    }
    for model in ("Count bigram", "Neural bigram", "MLP")
  ]


def test_comparison_requires_all_mandatory_models_and_finite_values():
  validate_comparison_rows(comparison_rows())
  with pytest.raises(ValueError, match="MLP"):
    validate_comparison_rows(comparison_rows()[:2])
  broken = comparison_rows()
  broken[0]["dev_nll"] = float("nan")
  with pytest.raises(ValueError, match="dev_nll"):
    validate_comparison_rows(broken)


def test_ablation_requires_three_settings_and_identical_controls():
  rows = [
    {
      "setting": setting,
      "controlled_variables": {"split": 42, "steps": 100},
      "independent_variable": "context_length",
      "result": 1.0,
    }
    for setting in (1, 2, 3)
  ]
  validate_ablation_rows(rows)
  with pytest.raises(ValueError, match="three"):
    validate_ablation_rows(rows[:2])
  broken = [row.copy() for row in rows]
  broken[-1]["controlled_variables"] = {"split": 7, "steps": 100}
  with pytest.raises(ValueError, match="controlled"):
    validate_ablation_rows(broken)
