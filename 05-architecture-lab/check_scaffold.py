"""Health checks for architecture-lab infrastructure, before TODOs."""

import importlib
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
LABS = {
  "01-baseline": "lab01_baseline",
  "02-component-change": "lab02_component_change",
  "03-ablation": "lab03_ablation",
  "04-report": "lab04_report",
}


def test_complete_lab_materials_exist():
  for lab, module in LABS.items():
    for name in ("README.md", "slides.md", "notes.md", f"{module}.py", f"check_{module}.py"):
      assert (PROJECT_DIR / lab / name).is_file(), f"missing {lab}/{name}"


def test_student_modules_import_without_executing_todos():
  for module in LABS.values():
    importlib.import_module(module)
