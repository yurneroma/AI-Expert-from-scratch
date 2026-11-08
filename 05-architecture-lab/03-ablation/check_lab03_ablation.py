"""Public control-variable, paired-seed, and summary checks."""

import pytest

from lab03_ablation import summarize_runs, validate_paired_runs


def run(variant, seed, dev_nll, tokens=1000):
  return {
    "variant": variant,
    "seed": seed,
    "config": {
      "tokenizer": "byte-v1",
      "data_split": "split-42",
      "training_tokens": 100_000,
      "effective_batch": 64,
      "optimizer": "adamw",
      "lr_schedule": "cosine-v1",
    },
    "metrics": {"dev_nll": dev_nll, "tokens_per_second": float(tokens)},
  }


def test_three_paired_seeds_are_valid_and_unshared_budget_fails():
  runs = [run(variant, seed, 1.0) for variant in ("baseline", "variant") for seed in (1, 2, 3)]
  assert validate_paired_runs(runs) is runs
  broken = [item.copy() for item in runs]
  broken[-1]["config"] = {**broken[-1]["config"], "training_tokens": 200_000}
  with pytest.raises(ValueError, match="training_tokens"):
    validate_paired_runs(broken)


def test_summary_uses_all_seeds_and_sample_standard_deviation():
  runs = [
    run("baseline", 1, 1.0, 1000),
    run("baseline", 2, 1.2, 1200),
    run("baseline", 3, 1.1, 1100),
    run("variant", 1, 1.1, 900),
    run("variant", 2, 1.0, 1000),
    run("variant", 3, 1.2, 1100),
  ]
  summary = summarize_runs(runs)
  assert summary["baseline"]["dev_nll"]["mean"] == pytest.approx(1.1)
  assert summary["baseline"]["dev_nll"]["std"] == pytest.approx(0.1)
  assert summary["variant"]["tokens_per_second"]["mean"] == pytest.approx(1000)
