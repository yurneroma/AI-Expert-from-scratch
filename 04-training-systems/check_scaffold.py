"""Health checks for training-systems infrastructure, before TODOs."""

import importlib
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
LABS = {
  "01-profiling": "lab01_profiling",
  "02-training-efficiency": "lab02_training_efficiency",
  "03-kv-cache": "lab03_kv_cache",
}


def test_complete_lab_materials_exist():
  for lab, module in LABS.items():
    for name in ("README.md", "slides.md", "notes.md", f"{module}.py", f"check_{module}.py"):
      assert (PROJECT_DIR / lab / name).is_file(), f"missing {lab}/{name}"


def test_student_modules_import_without_executing_todos():
  for module in LABS.values():
    importlib.import_module(module)
