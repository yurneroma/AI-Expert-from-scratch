"""Day 5 acceptance tests. Implement train_moon.py from top to bottom.

Run one stage at a time:

  uv run pytest -q 01-micrograd/test_train_moon.py -k dataset
  uv run pytest -q 01-micrograd/test_train_moon.py -k loss
  uv run pytest -q 01-micrograd/test_train_moon.py -k 'zero_grad or sgd'
  uv run pytest -q 01-micrograd/test_train_moon.py
"""

import random

import pytest

from engine import Value
from nn import MLP
from train_moon import loss, make_dataset, sgd_step, train, zero_grad


def test_dataset_has_expected_shape_and_labels():
  xs, ys = make_dataset(n_samples=20, noise=0.05, seed=7)

  assert len(xs) == len(ys) == 20
  assert all(len(x) == 2 for x in xs)
  assert set(ys) == {-1, 1}


class FixedModel:
  def __init__(self, scores, parameters=()):
    self.scores = iter(scores)
    self._parameters = list(parameters)

  def __call__(self, _):
    return [Value(next(self.scores))]

  def parameters(self):
    return self._parameters


def test_loss_matches_known_hinge_and_regularization_calculation():
  parameter = Value(2.0)
  model = FixedModel([0.5, -0.25], [parameter])

  result, accuracy = loss(model, [[0.0], [1.0]], [1, -1], alpha=0.1)

  # hinge mean = (0.5 + 0.75) / 2; regularization = 0.1 * 2**2
  assert result.data == pytest.approx(1.025)
  assert accuracy == 1.0


def test_zero_grad_clears_every_parameter():
  model = MLP(2, [2, 1])
  for parameter in model.parameters():
    parameter.grad = 3.0

  zero_grad(model)

  assert all(parameter.grad == 0.0 for parameter in model.parameters())


def test_sgd_step_updates_parameters_in_negative_gradient_direction():
  parameter = Value(2.0)
  parameter.grad = 0.5
  model = FixedModel([], [parameter])

  sgd_step(model, learning_rate=0.1)

  assert parameter.data == pytest.approx(1.95)


def test_training_reduces_loss_on_moons_and_reaches_target_accuracy():
  random.seed(42)
  xs, ys = make_dataset(n_samples=100, noise=0.1, seed=42)
  model = MLP(2, [16, 16, 1])

  history = train(model, xs, ys, epochs=100, learning_rate=1.0)

  assert len(history) == 100
  assert history[-1]["loss"] < history[0]["loss"]
  assert history[-1]["accuracy"] > 0.95
