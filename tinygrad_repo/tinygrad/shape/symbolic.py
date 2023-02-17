from __future__ import annotations
import math
from typing import List
from tinygrad.helpers import partition, modn, all_same

class Node:
  b, min, max = 0, -math.inf, math.inf   # make mypy happy
  def __str__(self):
    if self.min == self.max: return str(self.min)  # this is universal
    return self.expr
  @property
  def cl(self):
    return str(self).replace('//', '/')
  def __add__(self, b:int): return Variable.sum([self, Variable.num(b)])
  def __mul__(self, b:int):
    if b == 0: return NumNode(0)
    elif b == 1: return self
    return MulNode(self, b)
  def __floordiv__(self, b:int):
    assert b != 0
    if b == 1: return self
    if isinstance(self, MulNode) and self.b%b == 0: return self.a*(self.b//b)
    if isinstance(self, MulNode) and b%self.b == 0: return self.a//(b//self.b)
    if isinstance(self, SumNode):
      factors, tmp_nofactor = partition(self.nodes, lambda x: (isinstance(x, MulNode) or isinstance(x, NumNode)) and x.b%b == 0)
      nofactor = []
      # ugh, i doubt this is universally right
      for x in tmp_nofactor:
        if isinstance(x, NumNode):
          if modn(x.b, b) != x.b:
            factors.append(Variable.num(x.b - modn(x.b, b)))  # python does floor division
          nofactor.append(Variable.num(modn(x.b, b)))
        else:
          nofactor.append(x)
      gcd = [math.gcd(x.b, b) if isinstance(x, MulNode) or isinstance(x, NumNode) else None for x in nofactor]
      if len(factors) > 0:
        # these don't have to be the same, just having a common factor
        if len(gcd) > 0 and all_same(gcd) and gcd[0] is not None and gcd[0] > 1:
          nofactor_term = Variable.sum([(x.a * (x.b//gcd[0])) if isinstance(x, MulNode) else Variable.num(x.b//gcd[0]) for x in nofactor])//(b//gcd[0])
        else:
          nofactor_term = Variable.sum(nofactor)//b
        return Variable.sum([(x.a * (x.b//b)) if isinstance(x, MulNode) else Variable.num(x.b//b) for x in factors] + [nofactor_term])
    return DivNode(self, b)
  def __mod__(self, b:int):
    if b == 1: return NumNode(0)
    if isinstance(self, SumNode):
      a = Variable.sum([(x if not isinstance(x, NumNode) else Variable.num(modn(x.b, b))) for x in self.nodes if not (isinstance(x, MulNode) or isinstance(x, NumNode)) or (x.b%b != 0)])
    else:
      a = self
    if a.min >= 0 and a.max < b: return a
    return ModNode(a, b)
  def __ge__(self, b:int):
    if self.max < b: return Variable.num(0)
    if self.min >= b: return Variable.num(1)
    return GeNode(self, b)
  def __lt__(self, b:int):
    if self.max < b: return Variable.num(1)
    if self.min >= b: return Variable.num(0)
    return LtNode(self, b)

# constructors
class Variable(Node):
  def __init__(self, expr:str, nmin:int, nmax:int):
    self.expr, self.min, self.max = expr, nmin, nmax

  @staticmethod
  def num(num:int) -> Node:
    return NumNode(num)

  @staticmethod
  def sum(nodes:List[Node]) -> Node:
    if any([isinstance(x, SumNode) for x in nodes]):
      nodes, sum_nodes = partition(nodes, lambda x: not isinstance(x, SumNode))
      for x in sum_nodes: nodes += x.nodes
      return Variable.sum(nodes)
    nodes = [x for x in nodes if x.min != 0 or x.max != 0]
    if len(nodes) == 0: return NumNode(0)
    elif len(nodes) == 1: return nodes[0]
    return SumNode(nodes)

  @staticmethod
  def ands(nodes:List[Node]) -> Node:
    if any((x.min == 0 and x.max == 0) for x in nodes): return NumNode(0)
    nodes = [x for x in nodes if x.min != x.max]
    if len(nodes) == 0: return NumNode(1)
    elif len(nodes) == 1: return nodes[0]
    return AndNode(nodes)

class NumNode(Node):
  def __init__(self, num:int):
    self.b, self.min, self.max = num, num, num

class MulNode(Node):
  def __init__(self, a:Node, b:int):
    self.a, self.b = a, b
    self.min, self.max = a.min*b, a.max*b
  @property
  def expr(self):
    return f"({self.a}*{self.b})"

class DivNode(Node):
  def __init__(self, a:Node, b:int):
    self.a, self.b = a, b
    self.min, self.max = int(a.min/b), int(a.max/b)
  @property
  def expr(self):
    return f"({self.a}//{self.b})"

class ModNode(Node):
  def __init__(self, a:Node, b:int):
    self.a, self.b = a, b
    self.min, self.max = min(a.min, 0), max(a.max, b-1)
  @property
  def expr(self):
    assert self.a != self
    return f"({self.a}%{self.b})"

class GeNode(Node):
  def __init__(self, a:Node, b:int):
    self.a, self.b = a, b
    self.min, self.max = 0, 1
  @property
  def expr(self):
    return f"({self.a}>={self.b})"

class LtNode(Node):
  def __init__(self, a:Node, b:int):
    self.a, self.b = a, b
    self.min, self.max = 0, 1
  @property
  def expr(self):
    return f"({self.a}<{self.b})"

# reduce nodes

class SumNode(Node):
  def __init__(self, nodes:List[Node]):
    self.nodes = nodes
    self.min, self.max = sum([x.min for x in nodes]), sum([x.max for x in nodes])
  @property
  def expr(self):
    return f"({'+'.join([str(x) for x in self.nodes])})"

class AndNode(Node):
  def __init__(self, nodes:List[Node]):
    self.nodes = nodes
    self.min, self.max = min([x.min for x in nodes]), max([x.max for x in nodes])
  @property
  def expr(self):
    return f"({'&&'.join([str(x) for x in self.nodes])})"
