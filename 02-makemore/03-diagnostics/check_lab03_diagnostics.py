"""Public checks for Lab 03."""

import math

import pytest
import torch

from lab03_diagnostics import (
  gradient_statistics,
  recommended_weight_std,
  save_distribution_plot,
  tanh_saturation_fraction,
  tensor_statistics,
  update_to_data_ratio,
)


def test_weight_std_uses_fan_in_and_gain():
  assert recommended_weight_std(25) == pytest.approx(0.2)
  assert recommended_weight_std(25, gain=5 / 3) == pytest.approx(1 / 3)
  with pytest.raises(ValueError):
    recommended_weight_std(0)


def test_tensor_statistics_use_population_standard_deviation():
  result = tensor_statistics(torch.tensor([-1.0, 0.0, 1.0, 0.0]))
  assert result == pytest.approx({
    "mean": 0.0,
    "std": math.sqrt(0.5),
    "min": -1.0,
    "max": 1.0,
    "zero_fraction": 0.5,
  })


def test_saturation_fraction_uses_absolute_threshold():
  activations = torch.tensor([-0.99, -0.5, 0.98, 0.0])
  assert tanh_saturation_fraction(activations, threshold=0.97) == pytest.approx(0.5)


def test_gradient_statistics_requires_and_summarizes_gradients():
  parameter = torch.tensor([1.0, 2.0], requires_grad=True)
  (parameter**2).sum().backward()
  result = gradient_statistics({"p": parameter})
  assert result["p"]["mean"] == pytest.approx(3.0)
  with pytest.raises(ValueError):
    gradient_statistics({"missing": torch.ones(2, requires_grad=True)})


def test_update_ratio_compares_update_and_parameter_scale():
  parameter = torch.tensor([-1.0, 1.0], requires_grad=True)
  parameter.grad = torch.tensor([-0.1, 0.1])
  assert update_to_data_ratio(parameter, learning_rate=0.1) == pytest.approx(-2.0)


def test_distribution_plot_creates_nonempty_file(tmp_path):
  output = save_distribution_plot(
    {"a": torch.arange(10.0), "b": torch.arange(10.0) * 2},
    tmp_path / "distributions.png",
  )
  assert output == tmp_path / "distributions.png"
  assert output.is_file() and output.stat().st_size > 0

