"""Shared data utilities that are infrastructure, not lab solutions."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"


def default_data_path() -> Path:
  """Prefer the full local corpus and fall back to the bundled small corpus."""
  full_path = DATA_DIR / "names.txt"
  if full_path.exists():
    return full_path
  return DATA_DIR / "names_small.txt"


def load_words(path: str | Path | None = None) -> list[str]:
  """Load, normalize, and validate one lowercase word per line."""
  source = Path(path) if path is not None else default_data_path()
  words = [line.strip().lower() for line in source.read_text().splitlines()]
  words = [word for word in words if word]
  if not words:
    raise ValueError(f"dataset is empty: {source}")
  invalid = sorted({character for word in words for character in word if not character.isalpha()})
  if invalid:
    raise ValueError(f"dataset contains non-alphabetic characters: {invalid}")
  return words

