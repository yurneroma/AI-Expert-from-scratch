"""Health checks for Transformer course infrastructure, before TODOs."""

import importlib
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
LABS = {
  "01-tokenizer": "lab01_tokenizer",
  "02-attention": "lab02_attention",
  "03-transformer-block": "lab03_transformer_block",
  "04-training": "lab04_training",
  "05-generation": "lab05_generation",
  "06-reproducibility": "lab06_reproducibility",
}


def test_every_lab_has_reading_student_code_notes_and_checks():
  for lab, module in LABS.items():
    expected = ["README.md", "slides.md", "notes.md", f"{module}.py", f"check_{module}.py"]
    missing = [name for name in expected if not (PROJECT_DIR / lab / name).is_file()]
    assert not missing, f"{lab} missing: {missing}"


def test_student_modules_import_without_running_todos():
  for module in LABS.values():
    importlib.import_module(module)


def test_student_files_are_nonempty():
  for lab, module in LABS.items():
    assert (PROJECT_DIR / lab / f"{module}.py").stat().st_size > 500
