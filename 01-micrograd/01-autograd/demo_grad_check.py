from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

from engine import Value, grad_check

x = Value(2.0)
y = Value(3.0)

def f(x, y):
  return x * y + x

grad_check(f, [x, y])
print(x.grad)
print(y.grad)
