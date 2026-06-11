class Value:
  def __init__(self, data:float, children:tuple=(), op:str = ''):
    self._data = data
    self._prev = set(children)
    self._op = op
    self._backward = lambda: None

  def __add__(self, other:Value):
    out = Value(self._data + other._data, (self, other), '+')
    return out
  
  def __mul__(self, other:Value):
    out = Value(self._data * other._data, (self, other), '*')
    return out

  def __repr__(self):
    return f"Value(data={self._data}, children={self._children}, op={self._op})"

  