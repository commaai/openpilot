from __future__ import annotations
import math, itertools, functools
from typing import List, Dict, Callable, Type, Union
from tinygrad.helpers import partition, all_same

# NOTE: Python has different behavior for negative mod and floor div than c
# symbolic matches the Python behavior, but the code output is agnostic, and will never have negative numbers in div or mod

def create_node(typ:Type[Node], *args):
  ret = typ(*args)
  assert ret.min <= ret.max, f"min greater than max! {ret.min} {ret.max} when creating {typ} {args}"
  if ret.min == ret.max: return NumNode(ret.min)
  return ret

class Node:
  b: int
  min: int
  max: int
  def render(self, ops=None, ctx=None) -> str:
    if ops is None: ops = render_python
    assert isinstance(self, NumNode) or self.min != self.max
    return ops[type(self)](self, ops, ctx)
  @functools.cached_property
  def key(self) -> str: return self.render(ctx="DEBUG")
  def __repr__(self): return "<"+self.key+">"
  def __eq__(self, other:object) -> bool:
    if not isinstance(other, Node): return NotImplemented
    return self.key == other.key
  def __neg__(self): return self*-1
  def __add__(self, b:Union[Node, int]): return Variable.sum([self, b if isinstance(b, Node) else Variable.num(b)])
  def __sub__(self, b:Union[Node, int]): return self+-b
  def __ge__(self, b:int): return create_node(GeNode, self, b)
  def __lt__(self, b:int): return create_node(LtNode, self, b)
  def __mul__(self, b:int):
    if b == 0: return NumNode(0)
    elif b == 1: return self
    if isinstance(self, MulNode): return self.a*(self.b*b) # two muls is one mul
    if isinstance(self, SumNode): return Variable.sum([x*b for x in self.nodes]) # distribute mul into sum
    return create_node(MulNode, self, b)

  # *** complex ops ***

  def __floordiv__(self, b:int):
    assert b != 0
    if b < 0: return (self//-b)*-1
    if b == 1: return self
    if isinstance(self, DivNode): return self.a//(self.b*b) # two divs is one div
    if isinstance(self, MulNode) and self.b % b == 0: return self.a*(self.b//b)
    if isinstance(self, MulNode) and b % self.b == 0: return self.a//(b//self.b)
    if isinstance(self, SumNode):
      factors, tmp_nofactor = partition(self.nodes, lambda x: (isinstance(x, (MulNode, NumNode))) and x.b%b == 0)
      nofactor = []
      # ugh, i doubt this is universally right
      for x in tmp_nofactor:
        if isinstance(x, NumNode):
          if (x.b%b) != x.b:
            factors.append(Variable.num(x.b - (x.b%b)))  # python does floor division
          nofactor.append(Variable.num(x.b%b))
        else:
          nofactor.append(x)
      gcd = [math.gcd(x.b, b) if isinstance(x, (MulNode, NumNode)) else None for x in nofactor]
      if len(factors) > 0:
        # these don't have to be the same, just having a common factor
        if len(gcd) > 0 and all_same(gcd) and gcd[0] is not None and gcd[0] > 1:
          nofactor_term = Variable.sum([(x.a * (x.b//gcd[0])) if isinstance(x, MulNode) else Variable.num(x.b//gcd[0]) for x in nofactor])//(b//gcd[0])
        else:
          nofactor_term = Variable.sum(nofactor)//b
        return Variable.sum([(x.a * (x.b//b)) if isinstance(x, MulNode) else Variable.num(x.b//b) for x in factors] + [nofactor_term])
      else:
        muls = [x.b for x in nofactor if isinstance(x, MulNode)]
        for m in muls:
          if m > 1 and b%m == 0:
            return (self//m)//(b//m)
    if self.min < 0:
      offset = self.min//b
      return (self+offset*b)//b - offset
    return create_node(DivNode, self, b)

  def __mod__(self, b:int):
    assert b > 0
    if b == 1: return NumNode(0)
    if isinstance(self, SumNode):
      new_nodes = []
      for x in self.nodes:
        if isinstance(x, NumNode): new_nodes.append(Variable.num(x.b%b))
        elif isinstance(x, MulNode): new_nodes.append(x.a * (x.b%b))
        else: new_nodes.append(x)
      a = Variable.sum(new_nodes)
    elif isinstance(self, MulNode):
      a = self.a * (self.b%b)
    else:
      a = self
    if a.min >= 0 and a.max < b: return a
    if a.min < 0: return (a + ((a.min//b)*b)) % b
    return create_node(ModNode, a, b)

  @staticmethod
  def num(num:int) -> Node: return NumNode(num)

  @staticmethod
  def sum(nodes:List[Node]) -> Node:
    # expand any sums inside one sum
    if any([isinstance(x, SumNode) for x in nodes]):
      nodes, sum_nodes = partition(nodes, lambda x: not isinstance(x, SumNode))
      for x in sum_nodes: nodes += x.nodes
      return Variable.sum(nodes)

    # combine any numbers inside a sum
    nodes, num_nodes = partition(nodes, lambda x: not isinstance(x, NumNode))
    nodes.append(NumNode(sum([x.b for x in num_nodes])))

    # combine any MulNodes that factorize (big hack sticking the MulNode(x, 1) on things)
    nodes, mul_nodes = partition(nodes, lambda x: not isinstance(x, MulNode))
    mul_nodes += [MulNode(x, 1) for x in nodes]
    mul_nodes = sorted(mul_nodes, key=lambda x: x.a.render()) # group by equality (ugh, uses render!)
    new_nodes = [k * sum(x.b for x in g) for k, g in itertools.groupby(mul_nodes, key=lambda x: x.a)]
    nodes = [x if not isinstance(x, MulNode) or x.b != 1 else x.a for x in new_nodes]

    # filter 0s
    nodes = [x for x in nodes if x.min != 0 or x.max != 0]
    return create_node(SumNode, nodes) if len(nodes) > 1 else (nodes[0] if len(nodes) == 1 else NumNode(0))

  @staticmethod
  def ands(nodes:List[Node]) -> Node:
    if any((x.min == 0 and x.max == 0) for x in nodes): return NumNode(0)

    # filter 1s
    nodes = [x for x in nodes if x.min != x.max]
    return create_node(AndNode, nodes) if len(nodes) > 1 else (nodes[0] if len(nodes) == 1 else NumNode(1))

# 4 basic node types

class Variable(Node):
  def __new__(cls, expr:str, nmin:int, nmax:int):
    assert nmin >= 0 and nmin <= nmax
    if nmin == nmax: return NumNode(nmin)
    return super().__new__(cls)

  def __init__(self, expr:str, nmin:int, nmax:int):
    self.expr, self.min, self.max = expr, nmin, nmax

class NumNode(Node):
  def __init__(self, num:int):
    self.b, self.min, self.max = num, num, num

class OpNode(Node):
  def __init__(self, a:Node, b:int):
    self.a, self.b = a, b
    self.min, self.max = self.minmax(a,b)
  minmax = staticmethod(lambda a,b: (1//0, 1//0))

class RedNode(Node):
  def __init__(self, nodes:List[Node]):
    self.nodes = nodes
    self.min, self.max = self.minmax(nodes)
  minmax = staticmethod(lambda nodes: (1//0, 1//0))

# operation nodes

class GeNode(OpNode): minmax = staticmethod(lambda a,b: (int(a.min >= b), int(a.max >= b)))
class LtNode(OpNode): minmax = staticmethod(lambda a,b: (int(a.max < b), int(a.min < b)))
class MulNode(OpNode): minmax = staticmethod(lambda a,b: (a.min*b, a.max*b) if b >= 0 else (a.max*b, a.min*b))
class DivNode(OpNode):
  @staticmethod
  def minmax(a, b):
    assert a.min >= 0
    return a.min//b, a.max//b

class ModNode(OpNode):
  @staticmethod
  def minmax(a, b):
    assert a.min >= 0
    if a.max - a.min >= b or (a.min != a.max and a.min%b >= a.max%b): return (0, b-1)
    return a.min%b, a.max%b

# reduce nodes

class SumNode(RedNode): minmax = staticmethod(lambda nodes: (sum([x.min for x in nodes]), sum([x.max for x in nodes])))
class AndNode(RedNode): minmax = staticmethod(lambda nodes: (min([x.min for x in nodes]), max([x.max for x in nodes])))

render_python : Dict[Type, Callable] = {
  Variable: lambda self,ops,ctx: f"{self.expr}<{self.min},{self.max}>" if ctx == "DEBUG" else f"{self.expr}",
  NumNode: lambda self,ops,ctx: f"{self.b}",
  MulNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}*{self.b})",
  DivNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}//{self.b})",
  ModNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}%{self.b})",
  GeNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}>={self.b})",
  LtNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}<{self.b})",
  SumNode: lambda self,ops,ctx: f"({'+'.join(sorted([x.render(ops,ctx) for x in self.nodes]))})",
  AndNode: lambda self,ops,ctx: f"({'&&'.join(sorted([x.render(ops,ctx) for x in self.nodes]))})"
}