"""Expose the shared implementation to each independent lab's tests."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "shared"))
