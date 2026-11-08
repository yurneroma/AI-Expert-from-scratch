"""Lab 04: lightweight report and reproduction-contract validation."""

import re
from typing import Any


REQUIRED_SECTIONS = (
  "问题与假设",
  "实现证据",
  "实验协议",
  "结果",
  "解释与局限",
  "复现命令",
)
REQUIRED_REPRODUCTION_FIELDS = {
  "commit",
  "checkpoint",
  "train_command",
  "eval_command",
  "generate_command",
  "environment",
  "seeds",
}


def validate_report_markdown(markdown: str) -> None:
  """Require all report sections as nonempty level-two Markdown headings."""
  # TODO 1: report every missing section in one clear ValueError.
  raise NotImplementedError("TODO 1: implement validate_report_markdown")


def validate_reproduction_metadata(metadata: dict[str, Any]) -> None:
  """Require nonempty commands/artifacts plus environment, commit, and seeds."""
  # TODO 2: reject missing/empty fields and require at least three seeds.
  raise NotImplementedError("TODO 2: implement validate_reproduction_metadata")
