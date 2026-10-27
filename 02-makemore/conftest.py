"""Expose lab modules and shared data helpers during explicit course checks."""

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
for directory in [PROJECT_DIR / "shared", *sorted(PROJECT_DIR.glob("[0-9][0-9]-*"))]:
  if directory.is_dir() and str(directory) not in sys.path:
    sys.path.insert(0, str(directory))
