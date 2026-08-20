"""Day 5 lab: train the scalar MLP on a two-moons dataset.

Complete the TODOs in order. NumPy, scikit-learn, and matplotlib may be used
for data generation and plotting, but the model, loss, and optimization must
use the Value/MLP implementation from this project.
"""

from pathlib import Path
import random

from nn import MLP


def make_dataset(n_samples=100, noise=0.1, seed=42):
  """Return moon inputs and labels, with labels encoded as -1 or +1.

  Returns:
    xs: list[list[float]], each input has two coordinates.
    ys: list[int], each label is either -1 or +1.
  """
  from sklearn.datasets import make_moons
  xs, ys = make_moons(
    n_samples=n_samples,
    noise=noise,
    random_state=seed,
  )
  xs = xs.tolist()
  ys = [1 if label ==1 else  -1 for label in ys]
  return xs, ys 


def save_dataset_plot(xs, ys, output_dir=None):
  """Draw the two classes in the original dataset and save the figure."""
  import matplotlib.pyplot as plt
  import numpy as np

  output_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "moons_dataset.png"

  points = np.asarray(xs)
  labels = np.asarray(ys)

  figure, axis = plt.subplots(figsize=(8, 5))
  for label, color in [(-1, "royalblue"), (1, "tomato")]:
    class_points = points[labels == label]
    axis.scatter(
      class_points[:, 0],
      class_points[:, 1],
      color=color,
      edgecolors="black",
      label=f"class {label:+d}",
    )

  axis.set_title("Original make_moons dataset")
  axis.set_xlabel("x1")
  axis.set_ylabel("x2")
  axis.legend()
  axis.set_aspect("equal", adjustable="datalim")
  figure.tight_layout()
  figure.savefig(output_path)
  plt.close(figure)
  return output_path


def loss(model, xs, ys, alpha=1e-4):
  """Compute batch hinge loss with L2 regularization.

  For each example, let score be the model's single output and compute
  max(0, 1 - y * score). Average those losses, then add
  alpha * sum(parameter**2).

  Returns:
    (total_loss, accuracy), where total_loss is a Value and accuracy is float.
  """
  # Run all forward passes and keep the scalar score from each MLP call.
  scores = [model(x)[0] for x in xs]
  # Build the mean hinge loss using Value operations.
  from nn import Value
  losses = [
    (Value(1) - y * score)
    for y, score in zip(ys, scores)
  ]
  losses = [l.relu() for l in losses]
  data_loss = sum(losses) / len(losses)

  # Add L2 regularization and calculate classification accuracy.
  reg_loss = alpha * sum(
    parameter**2 for parameter in model.parameters()
  )
  total_loss = data_loss + reg_loss
  
  accuracy = sum(
    (score.data > 0) == (y > 0) 
    for y, score in zip(ys, scores)
    ) / len(ys)


  return total_loss, accuracy

  


def zero_grad(model):
  """Reset every model parameter's gradient before backpropagation."""
  # Set each parameter gradient to zero.
  for parameter in model.parameters():
    parameter.grad = 0.0 


def sgd_step(model, learning_rate):
  """Apply one in-place SGD parameter update."""
  for parameter in model.parameters():
    parameter.data -= learning_rate * parameter.grad


def train(model, xs, ys, epochs=100, learning_rate=1.0, alpha=1e-4):
  """Train with full-batch gradient descent and return epoch history.

  Each history item must be a dict containing epoch, loss, and accuracy.
  Use a linearly decaying learning rate from learning_rate to 10% of it.
  """
  history = []
  for epoch in range(epochs):
    #1. forward 
    total_loss, accuracy = loss(model, xs, ys, alpha)
    #2. reset grad of last epoch 
    zero_grad(model)

    #3. backward
    total_loss.backward()

    #4. update parameter 
    current_learning_rate = learning_rate * (
      1 - 0.9 *epoch/epochs
    )
    sgd_step(model, current_learning_rate)

    #5. record
    history.append({
      "epoch":epoch,
      "loss":total_loss.data,
      "accuracy":accuracy,
    })
  return history


def save_plots(model, xs, ys, history, output_dir=None):
  """Save loss_curve.png and decision_boundary.png, returning both paths."""
  import matplotlib.pyplot as plt
  import numpy as np

  output_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
  output_dir.mkdir(parents=True, exist_ok=True)
  loss_path = output_dir / "loss_curve.png"

  figure, axis = plt.subplots()
  axis.plot(
    [item["epoch"] for item in history],
    [item["loss"] for item in history],
  )
  axis.set_xlabel("Epoch")
  axis.set_ylabel("Loss")
  figure.tight_layout()
  figure.savefig(loss_path)
  plt.close(figure)

  points = np.asarray(xs)
  padding = 0.5
  step = 0.025
  x_min, x_max = points[:, 0].min() - padding, points[:, 0].max() + padding
  y_min, y_max = points[:, 1].min() - padding, points[:, 1].max() + padding
  grid_x, grid_y = np.meshgrid(
    np.arange(x_min, x_max, step),
    np.arange(y_min, y_max, step),
  )
  grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
  predictions = np.fromiter(
    (model(point)[0].data > 0 for point in grid_points),
    dtype=bool,
    count=len(grid_points),
  ).reshape(grid_x.shape)

  boundary_path = output_dir / "decision_boundary.png"
  figure, axis = plt.subplots()
  axis.contourf(grid_x, grid_y, predictions, levels=[-0.5, 0.5, 1.5], alpha=0.35)
  axis.scatter(points[:, 0], points[:, 1], c=ys, cmap="coolwarm", edgecolors="black")
  axis.set_xlabel("x1")
  axis.set_ylabel("x2")
  figure.tight_layout()
  figure.savefig(boundary_path)
  plt.close(figure)
  return loss_path, boundary_path


def main():
  """Train the Day 5 model and write both required figures."""
  random.seed(42)
  xs, ys = make_dataset()
  dataset_path = save_dataset_plot(xs, ys)
  model = MLP(2, [16, 16, 1])
  history = train(model, xs, ys)
  loss_path, boundary_path = save_plots(model, xs, ys, history)

  final = history[-1]
  print(f"Final loss: {final['loss']:.4f}")
  print(f"Final accuracy: {final['accuracy']:.2%}")
  print(f"Dataset: {dataset_path}")
  print(f"Loss curve: {loss_path}")
  print(f"Decision boundary: {boundary_path}")


if __name__ == "__main__":
  main()
