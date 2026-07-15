from pathlib import Path

from engine import Value
from visualize import draw_dot

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

e = a * b
e.label = 'e'
d = e + c
d.label = 'd'

out = d
out.label = 'out'
out.backward()

dot = draw_dot(out)
output_path = Path(__file__).with_name('graph')
rendered_path = dot.render(str(output_path), cleanup=True)
print(f'graph saved to: {rendered_path}')
