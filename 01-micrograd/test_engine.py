from engine import Value

def test_add():
  a = Value(2.0)
  b = Value(3.0)
  c = a + b
  assert c._data == 5.0
  assert c._prev == {a, b}
  assert c._op == '+'

def test_mul():
  a = Value(2.0)
  b = Value(3.0)
  c = a * b
  assert c._data == 6.0

def test_children(): 
  a = Value(2.0)
  b = Value(3.0)
  c = a + b
  assert c._prev == {a, b}

def test_op():
  a = Value(2.0)
  b = Value(3.0)
  c = a * b
  assert c._op == '*'

def test_self_reuse():
  a = Value(2.0)
  b = a + a
  assert b._prev == {a}
  assert b._op == '+'
