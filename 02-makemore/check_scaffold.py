"""Health checks for the provided course infrastructure.

This file should pass before any student TODO is implemented.
"""

import importlib
from pathlib import Path

import torch

from data import default_data_path, load_words


PROJECT_DIR = Path(__file__).resolve().parent
LABS = {
  "01-bigram": "lab01_bigram",
  "02-mlp": "lab02_mlp",
  "03-diagnostics": "lab03_diagnostics",
  "04-batchnorm": "lab04_batchnorm",
  "05-backprop": "lab05_backprop",
  "06-wavenet": "lab06_wavenet",
}


def test_required_course_material_exists():
  expected = [
    "schedule.md",
    *[f"{lab}/{name}" for lab in LABS for name in ["README.md", "slides.md", "notes.md"]],
    "07-capstone/README.md",
    "07-capstone/notes.md",
  ]
  missing = [path for path in expected if not (PROJECT_DIR / path).is_file()]
  assert not missing, f"missing course files: {missing}"


def test_student_modules_import_without_executing_todos():
  for module in LABS.values():
    importlib.import_module(module)


def test_bundled_dataset_is_available_offline():
  words = load_words(PROJECT_DIR / "data" / "names_small.txt")
  assert default_data_path().is_file()
  assert len(words) >= 50
  assert all(word.isalpha() and word.islower() for word in words)


def test_runtime_has_required_pytorch_features():
  generator = torch.Generator().manual_seed(42)
  values = torch.randn(4, generator=generator)
  assert values.shape == (4,)
  assert torch.isfinite(values).all()


def test_student_files_exist_and_are_nonempty():
  student_files = [PROJECT_DIR / lab / f"{module}.py" for lab, module in LABS.items()]
  assert len(student_files) == 6
  assert all(path.stat().st_size > 500 for path in student_files)
