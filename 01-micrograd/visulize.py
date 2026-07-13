from graphviz import Digraph

class Visualize:
  def __init__(self):
    self.graph = Digraph()
    self.graph.attr(rankdir='LR')

  def add_node(self, node):
    self.graph.node(node.name, node.data, node.lable)

  def add_edge(self, node1, node2):
    self.graph.edge(node1.name, node2.name)

  def visualize(self):
    self.graph.render('graph', format='png')


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
  vis = Visualize()
  nodes, edges = trace(root)
  for n in nodes:
    vis.add_node(n)
  for n1, n2 in edges:
    vis.add_edge()
  vis.render('graph', format='png')
  return vis