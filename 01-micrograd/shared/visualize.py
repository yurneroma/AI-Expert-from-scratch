from graphviz import Digraph


def trace(root):
  nodes, edges = set(), set()

  def build(v):
    if v in nodes:
      return
    nodes.add(v)
    for child in v._prev:
      edges.add((child, v))
      build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
  nodes, edges = trace(root)

  for value in nodes:
    value_id = f"value_{id(value)}"
    dot.node(
      name=value_id,
      label=(
        f"{{ {value.label} | "
        f"data {value.data:.4f} | "
        f"grad {value.grad:.4f} }}"
      ),
      shape='record',
    )

    if value._op:
      op_id = f"op_{id(value)}"
      dot.node(name=op_id, label=value._op)
      dot.edge(op_id, value_id)

  for source, result in edges:
    source_id = f"value_{id(source)}"
    result_op_id = f"op_{id(result)}"
    dot.edge(source_id, result_op_id)

  return dot
