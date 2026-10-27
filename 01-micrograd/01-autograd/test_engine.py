from engine import Value, grad_check

def test_add():
  a = Value(2.0)
  b = Value(3.0)
  c = a + b
  assert c.data == 5.0
  assert c._prev == {a, b}
  assert c._op == '+'

def test_mul():
  a = Value(2.0)
  b = Value(3.0)
  c = a * b
  assert c.data == 6.0

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

def test_repr():
  a = Value(2.0)
  assert str(a) == "Value(data=2.0, children=[], op='')"

def test_backward_simple_add():
  a = Value(2.0)
  b = Value(3.0)
  c = a + b
  c.backward()
  assert a.grad == 1.0
  assert b.grad == 1.0

def test_backward_simple_mul():
  a = Value(2.0)
  b = Value(3.0)
  c = a * b
  c.backward()
  assert a.grad == 3.0
  assert b.grad == 2.0

def test_backward_simple_pow():
  a = Value(2.0)
  c = a ** 3
  c.backward()
  assert a.grad == 12.0

def test_backward_simple_neg():
  a = Value(2.0)
  c = -a
  c.backward()
  assert a.grad == -1.0

def test_backward_simple_sub():
  a = Value(2.0)
  b = Value(3.0)
  c = a - b
  c.backward()
  assert a.grad == 1.0
  assert b.grad == -1.0

def test_backward_simple_div():
  a = Value(2.0)
  b = Value(3.0)
  c = a / b
  c.backward() 
  assert a.grad == 1/3 
  assert b.grad == -2/9

def test_backward_expression():
  a = Value(2.0)
  b = Value(-3.0)
  c = Value(10.0)
  d = a*b+c
  d.backward()
  assert a.grad == -3.0
  assert b.grad == 2.0
  assert c.grad == 1.0

def test_grad_check_expression():
  x = Value(2.0)
  y = Value(3.0)

  assert grad_check(lambda x, y: x * y + x, [x, y])
  assert x.grad == 4.0
  assert y.grad == 2.0
 
