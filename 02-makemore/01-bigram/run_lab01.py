"""Run the required Lab 01 comparison after all student TODOs pass."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "shared"))

from data import default_data_path, load_words  # noqa: E402
from lab01_bigram import (  # noqa: E402
  average_nll,
  build_bigram_counts,
  build_bigram_dataset,
  build_vocab,
  counts_to_probabilities,
  sample_bigram,
  train_neural_bigram,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", type=Path, default=default_data_path())
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument("--learning-rate", type=float, default=20.0)
  parser.add_argument("--regularization", type=float, default=0.01)
  parser.add_argument("--seed", type=int, default=2147483647)
  parser.add_argument("--samples", type=int, default=10)
  parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parent / "artifacts" / "metrics.json",
  )
  return parser.parse_args()


def generated_samples(
    probabilities: torch.Tensor,
    stoi: dict[str, int],
    itos: dict[int, str],
    seed: int,
    count: int,
) -> list[str]:
  generator = torch.Generator().manual_seed(seed)
  return sample_bigram(
    probabilities,
    stoi,
    itos,
    generator,
    n_samples=count,
  )


def main() -> None:
  args = parse_args()
  words = load_words(args.data)
  stoi, itos = build_vocab(words)
  counts = build_bigram_counts(words, stoi)
  inputs, targets = build_bigram_dataset(words, stoi)

  count_results = []
  for smoothing in (0.1, 1.0, 10.0):
    probabilities = counts_to_probabilities(counts, smoothing=smoothing)
    count_results.append({
      "smoothing": smoothing,
      "nll": average_nll(words, probabilities, stoi),
      "samples": generated_samples(
        probabilities, stoi, itos, args.seed, args.samples
      ),
    })

  weights, history = train_neural_bigram(
    inputs,
    targets,
    vocab_size=len(stoi),
    steps=args.steps,
    learning_rate=args.learning_rate,
    regularization=args.regularization,
    seed=args.seed,
  )
  neural_probabilities = torch.softmax(weights, dim=1)
  neural_result = {
    "steps": args.steps,
    "learning_rate": args.learning_rate,
    "regularization": args.regularization,
    "initial_objective": history[0],
    "final_objective": history[-1],
    "nll": average_nll(words, neural_probabilities, stoi),
    "samples": generated_samples(
      neural_probabilities, stoi, itos, args.seed, args.samples
    ),
  }

  result = {
    "data": str(args.data),
    "word_count": len(words),
    "vocab_size": len(stoi),
    "seed": args.seed,
    "uniform_nll": math.log(len(stoi)),
    "count_models": count_results,
    "neural_bigram": neural_result,
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + "\n")

  print(f"words={len(words):,} vocab={len(stoi)} uniform_nll={result['uniform_nll']:.4f}")
  for item in count_results:
    print(f"count smoothing={item['smoothing']:>4}: nll={item['nll']:.4f}")
  print(f"neural bigram: nll={neural_result['nll']:.4f}")
  print("samples:", ", ".join(neural_result["samples"]))
  print(f"saved: {args.output}")


if __name__ == "__main__":
  main()
