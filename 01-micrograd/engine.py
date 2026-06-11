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
    return out
  
  def __mul__(self, other:Value):
    out = Value(self.data * other.data, (self, other), '*')
    return out

  def __repr__(self):
    return f"Value(data={self.data}, children={list(self._prev)}, op={self._op!r})"

  