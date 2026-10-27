"""Download the full educational names corpus used by makemore."""

from pathlib import Path
from urllib.request import urlopen


URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "names.txt"


def main() -> None:
  """Fetch to a temporary file, validate basic structure, then replace output."""
  print(f"Downloading {URL}")
  payload = urlopen(URL, timeout=30).read()
  words = payload.decode("utf-8").splitlines()
  if len(words) < 30_000 or not all(word.isalpha() and word.islower() for word in words):
    raise RuntimeError("downloaded corpus failed validation")

  OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  temporary = OUTPUT.with_suffix(".tmp")
  temporary.write_bytes(payload)
  temporary.replace(OUTPUT)
  print(f"Saved {len(words):,} names to {OUTPUT}")


if __name__ == "__main__":
  main()

