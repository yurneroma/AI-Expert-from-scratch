from __future__ import annotations
class Value:
  def __init__(self, data:float, children:tuple=(), op:str=''):
    self.data = data
    self._prev = set(children)
    self._op = op
    self._backward = lambda: None
    self.grad = 0.0

  def __add__(self, other:Value):
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad += 1.0*out.grad
      other.grad  += 1.0*out.grad

    out._backward = _backward
    return out
  
  def __mul__(self, other:Value):
    out = Value(self.data * other.data, (self, other), '*')
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self, power):
    assert isinstance(power, (int, float)), "only supports int/float powers"
    out = Value(self.data**power, (self, ), f'**{power}')
    def _backward():
      self.grad += power * self.data ** (power - 1) * out.grad
    out._backward = _backward
    return out

  def __neg__(self):
    out = Value(-self.data, (self, ), '-')
    def _backward():
      self.grad += -1.0 * out.grad
    out._backward = _backward
    return out

  def __sub__(self, other:Value):
    return self + (-other)

  def __truediv__(self, other:Value):
    return self * other**-1

  def __radd__(self, other):
    pass


  def __repr__(self):
    return f"Value(data={self.data}, children={list(self._prev)}, op={self._op!r})"

  def backward(self): 
    self.grad = 1.0
    topo = []
    visited = set()

    def build_topo(v):
      """
        Builds a topological ordering of the nodes in the graph ending at v.
      """
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    
    build_topo(self)

    # backpropagate the gradients through the graph
    for node in reversed(topo):
      node._backward()





  
  