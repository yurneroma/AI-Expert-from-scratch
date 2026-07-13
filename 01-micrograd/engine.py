from __future__ import annotations
import math
class Value:
  def __init__(self, data:float, children:tuple=(), op:str=''):
    self.data = data
    self._prev = set(children)
    self._op = op
    self._backward = lambda: None
    self.grad = 0.0

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad += 1.0*out.grad
      other.grad  += 1.0*out.grad

    out._backward = _backward
    return out
  
  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
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
    return self + other

  def __rmul__(self, other):
    return self * other


  def __repr__(self):
    return f"Value(data={self.data}, children={list(self._prev)}, op={self._op!r})"

  def backward(self): 

    """
      由于每个 Value 对象都保存了它的子节点和操作符，因此我们可以通过反向遍历这个图来计算梯度。
      每个Value 节点调用_backward(), 把局部系数传递给它的输入节点。
      所以我们只需要构建一个拓扑排序，然后反向遍历这个图，计算每个节点的梯度。
    """
    self.grad = 1.0
    topo = []
    visited = set()

    def build_topo(v):
      """
        构建拓扑排序
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


  def tanh(self):
    t = math.tanh(self.data)
    out = Value(t, (self, ), 'tanh')
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    return out
  def relu(self):
    r = max(0, self.data)
    out = Value(r, (self, ), 'relu')
    def _backward():
      self.grad += (r > 0) * out.grad
    out._backward = _backward
    return out

  def exp(self):
    e = math.exp(self.data)
    out = Value(e, (self, ), 'exp')
    def _backward():
      self.grad += e * out.grad
    out._backward = _backward
    return out

  def log(self):
    l = math.log(self.data)
    out = Value(l, (self, ), 'log')
    def _backward():
      self.grad += 1/self.data * out.grad
    out._backward = _backward
    return out

  

  
  