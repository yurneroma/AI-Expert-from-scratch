"""Public checks that reports retain evidence and reproduction contracts."""

import pytest

from lab04_report import validate_report_markdown, validate_reproduction_metadata


def complete_report():
  return "\n".join(f"## {section}\ncontent" for section in (
    "问题与假设", "实现证据", "实验协议", "结果", "解释与局限", "复现命令"
  ))


def metadata():
  return {
    "commit": "abc123",
    "checkpoint": "artifacts/model.pt",
    "train_command": "uv run python train.py --resume",
    "eval_command": "uv run python eval.py",
    "generate_command": "uv run python generate.py",
    "environment": "Python / torch / device recorded",
    "seeds": [1, 2, 3],
  }


def test_complete_report_and_missing_sections():
  validate_report_markdown(complete_report())
  with pytest.raises(ValueError, match="结果"):
    validate_report_markdown("## 问题与假设\ncontent")


def test_reproduction_metadata_requires_commands_and_three_seeds():
  validate_reproduction_metadata(metadata())
  incomplete = metadata()
  incomplete["seeds"] = [1]
  with pytest.raises(ValueError, match="seeds"):
    validate_reproduction_metadata(incomplete)
  incomplete = metadata()
  incomplete["eval_command"] = ""
  with pytest.raises(ValueError, match="eval_command"):
    validate_reproduction_metadata(incomplete)
