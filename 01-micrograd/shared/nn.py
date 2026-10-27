"""Day 4 lab: build neural-network modules from scalar Value objects.

Complete the TODOs in this file in order:

1. Neuron
2. Layer
3. MLP

Do not use NumPy or PyTorch in this file. Every parameter must be a Value so
that engine.py can build the computation graph and run backward propagation.
"""

from engine import Value
import random

class Neuron:
  """A single tanh neuron: tanh(w dot x + b)."""

  def __init__(self, nin: int):
    """Create one weight per input and one bias.

    Args:
      nin: Number of scalar inputs accepted by this neuron.

    Required attributes:
      self.w: list[Value] with length nin.
      self.b: one Value.

    Initialization requirement:
      Initialize every parameter randomly in the interval [-1, 1].
    """
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self, x):
    """Run the neuron forward pass.

    Args:
      x: Sequence of nin numbers or Value objects.

    Returns:
      A Value representing tanh(w dot x + b).

    Mathematical target:
      activation = w[0] * x[0] + ... + w[n-1] * x[n-1] + b
      output = tanh(activation)
    """
    activation = self.b
    for weight, input_value in zip(self.w, x):
      activation = activation + weight * input_value
    return activation.tanh()

  def parameters(self):
    """Return all trainable parameters as one flat list of Value objects."""
    return self.w + [self.b]


class Layer:
  """A group of neurons that all receive the same input vector."""

  def __init__(self, nin: int, nout: int):
    """Create nout independent Neuron objects, each with nin inputs.

    Required attribute:
      self.neurons: list[Neuron] with length nout.
    """
    self.neurons = [Neuron(nin) for _ in range(nout)]


  def __call__(self, x):
    """Evaluate every neuron using the same input x.

    Returns:
      list[Value] with one output per neuron. Always return a list, including
      when the layer contains exactly one neuron.
    """
    return [neuron(x) for neuron in self.neurons]

  def parameters(self):
    """Return all neuron parameters as one flat list."""
    return [parameter for neuron in self.neurons  for parameter in neuron.parameters() ]


class MLP:
  """A multilayer perceptron made by chaining Layer objects."""

  def __init__(self, nin: int, nouts: list[int]):
    """Build one Layer for each requested output width.

    Example:
      MLP(3, [4, 4, 1]) creates Layer(3, 4), Layer(4, 4), Layer(4, 1).

    Required attribute:
      self.layers: list[Layer] with length len(nouts).
    """
    self.layers = [Layer(nin, nout) for nin, nout in zip([nin] + nouts, nouts)]

  def __call__(self, x):
    """Pass x through every layer in order.

    Returns:
      list[Value] produced by the final layer.
    """
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    """Return every trainable parameter in the network as one flat list."""
    return [parameter for layer in self.layers for parameter in layer.parameters()]
