from __future__ import annotations
from abc import abstractmethod
import functools
from math import gcd
from itertools import product
from tinygrad.helpers import partition
from typing import List, Dict, Callable, Tuple, Type, Union, Optional, Any, Iterator

# NOTE: Python has different behavior for negative mod and floor div than c
# symbolic matches the Python behavior, but the code output is agnostic, and will never have negative numbers in div or mod

def is_sym_int(x: Any) -> bool: return isinstance(x, (int, Node))

class Node:
  b: Union[Node, int]
  min: int
  max: int
  def render(self, ops=None, ctx=None) -> Any:
    if ops is None: ops = render_python
    assert self.__class__ in (Variable, NumNode) or self.min != self.max
    return ops[type(self)](self, ops, ctx)
  def vars(self): return []

  def expand_idx(self) -> VariableOrNum: return next((v for v in self.vars() if v.expr is None), NumNode(0))
  # expand a Node into List[Node] that enumerates the underlying Variables from min to max
  # expand increments earlier variables faster than later variables (as specified in the argument)
  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def expand(self, idxs:Optional[Tuple[VariableOrNum, ...]]=None) -> List[Node]:
    if idxs is None: idxs = (self.expand_idx(),)
    return [self.substitute(dict(zip(idxs, (NumNode(x) for x in rep)))) for rep in Node.iter_idxs(idxs)]
  @staticmethod
  def iter_idxs(idxs:Tuple[VariableOrNum, ...]) -> Iterator[Tuple[int,...]]:
    yield from (x[::-1] for x in product(*[[x for x in range(v.min, v.max + 1)] for v in idxs[::-1]]))
  # substitute Variables with the values in var_vals
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: raise RuntimeError(self.__class__.__name__)
  def unbind(self) -> Tuple[Node, Optional[int]]: return self.substitute({v: v.unbind()[0] for v in self.vars() if v.val is not None}), None

  @functools.cached_property
  def key(self) -> str: return self.render(ctx="DEBUG")
  @functools.cached_property
  def hash(self) -> int: return hash(self.key)
  def __repr__(self): return self.render(ctx="REPR")
  def __str__(self): return "<"+self.key+">"
  def __hash__(self): return self.hash
  def __bool__(self): return not (self.max == self.min == 0)
  def __eq__(self, other:object) -> bool:
    if not isinstance(other, Node): return NotImplemented
    return self.key == other.key
  def __neg__(self): return self*-1
  def __add__(self, b:Union[Node,int]): return Variable.sum([self, b if isinstance(b, Node) else Variable.num(b)])
  def __radd__(self, b:int): return self+b
  def __sub__(self, b:Union[Node,int]): return self+-b
  def __rsub__(self, b:int): return -self+b
  def __le__(self, b:Union[Node,int]): return self < (b+1)
  def __gt__(self, b:Union[Node,int]): return (-self) < (-b)
  def __ge__(self, b:Union[Node,int]): return (-self) < (-b+1)
  def __lt__(self, b:Union[Node,int]): return create_node(LtNode(self, b))
  def __mul__(self, b:Union[Node, int]):
    if b == 0: return NumNode(0)
    if b == 1: return self
    if self.__class__ is NumNode: return NumNode(self.b*b) if isinstance(b, int) else b*self.b
    return create_node(MulNode(self, b.b)) if isinstance(b, NumNode) else create_node(MulNode(self, b))
  def __rmul__(self, b:int): return self*b

  # *** complex ops ***

  def __rfloordiv__(self, b:int):
    if self.min > b >= 0: return NumNode(0)
    if isinstance(self, NumNode): return NumNode(b // self.b)
    raise RuntimeError(f"not supported: {b} // {self}")
  def __floordiv__(self, b:Union[Node,int], factoring_allowed=True):
    if isinstance(b, Node):
      if b.__class__ is NumNode: return self // b.b
      if self == b: return NumNode(1)
      if (b - self).min > 0 and self.min >= 0: return NumNode(0) # b - self simplifies the node
      raise RuntimeError(f"not supported: {self} // {b}")
    assert b != 0
    if b < 0: return (self//-b)*-1
    if b == 1: return self

    # the numerator of div is not allowed to be negative
    if self.min < 0:
      offset = self.min//b
      # factor out an "offset" to make the numerator positive. don't allowing factoring again
      return (self + -offset*b).__floordiv__(b, factoring_allowed=False) + offset
    return create_node(DivNode(self, b))

  def __rmod__(self, b:int):
    if self.min > b >= 0: return NumNode(b)
    if isinstance(self, NumNode): return NumNode(b % self.b)
    raise RuntimeError(f"not supported: {b} % {self}")
  def __mod__(self, b:Union[Node,int]):
    if isinstance(b, Node):
      if b.__class__ is NumNode: return self % b.b
      if self == b: return NumNode(0)
      if (b - self).min > 0 and self.min >= 0: return self # b - self simplifies the node
      raise RuntimeError(f"not supported: {self} % {b}")
    assert b > 0
    if b == 1: return NumNode(0)
    if self.min >= 0 and self.max < b: return self
    if (self.min//b) == (self.max//b): return self - (b*(self.min//b))
    if self.min < 0: return (self - ((self.min//b)*b)) % b
    return create_node(ModNode(self, b))

  @staticmethod
  def num(num:int) -> NumNode: return NumNode(num)

  @staticmethod
  def factorize(nodes:List[Node]) -> List[Node]:
    mul_groups: Dict[Node, int] = {}
    for x in nodes:
      a,b = (x.a,x.b) if isinstance(x, MulNode) else (x,1)
      mul_groups[a] = mul_groups.get(a, 0) + b
    return [MulNode(a, b_sum) if b_sum != 1 else a for a, b_sum in mul_groups.items() if b_sum != 0]

  @staticmethod
  def sum(nodes:List[Node]) -> Node:
    nodes = [x for x in nodes if x.max or x.min]
    if not nodes: return NumNode(0)
    if len(nodes) == 1: return nodes[0]

    new_nodes: List[Node] = []
    num_node_sum = 0
    for node in SumNode(nodes).flat_components:
      if node.__class__ is NumNode: num_node_sum += node.b
      else: new_nodes.append(node)

    if len(new_nodes) > 1 and len(set([x.a if isinstance(x, MulNode) else x for x in new_nodes])) < len(new_nodes):
      new_nodes = Node.factorize(new_nodes)
    if num_node_sum: new_nodes.append(NumNode(num_node_sum))
    return create_rednode(SumNode, new_nodes) if len(new_nodes) > 1 else new_nodes[0] if len(new_nodes) == 1 else NumNode(0)

  @staticmethod
  def ands(nodes:List[Node]) -> Node:
    if not nodes: return NumNode(1)
    if len(nodes) == 1: return nodes[0]
    if any(not x for x in nodes): return NumNode(0)

    # filter 1s
    nodes = [x for x in nodes if x.min != x.max]
    return create_rednode(AndNode, nodes) if len(nodes) > 1 else (nodes[0] if len(nodes) == 1 else NumNode(1))

# 4 basic node types

class Variable(Node):
  def __new__(cls, expr:Optional[str], nmin:int, nmax:int):
    assert nmin >= 0 and nmin <= nmax
    if nmin == nmax: return NumNode(nmin)
    return super().__new__(cls)

  def __init__(self, expr:Optional[str], nmin:int, nmax:int):
    self.expr, self.min, self.max = expr, nmin, nmax
    self.val:Optional[int] = None
  def bind(self, val):
    assert self.val is None and self.min<=val<=self.max, f"cannot bind {val} to {self}"
    self.val = val
    return self
  def unbind(self) -> Tuple[Variable, int]:
    assert self.val is not None, f"cannot unbind {self}"
    return Variable(self.expr, self.min, self.max), self.val
  def vars(self): return [self]
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: return var_vals[self] if self in var_vals else self

class NumNode(Node):
  def __init__(self, num:int):
    assert isinstance(num, int), f"{num} is not an int"
    self.b:int = num
    self.min, self.max = num, num
  def bind(self, val):
    assert self.b == val, f"cannot bind {val} to {self}"
    return self
  def __eq__(self, other): return self.b == other
  def __hash__(self): return self.hash  # needed with __eq__ override
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: return self

def create_node(ret:Node):
  assert ret.min <= ret.max, f"min greater than max! {ret.min} {ret.max} when creating {type(ret)} {ret}"
  if ret.min == ret.max: return NumNode(ret.min)
  return ret

class OpNode(Node):
  def __init__(self, a:Node, b:Union[Node, int]):
    self.a, self.b = a, b
    self.min, self.max = self.get_bounds()
  def vars(self): return self.a.vars() + (self.b.vars() if isinstance(self.b, Node) else [])
  @abstractmethod
  def get_bounds(self) -> Tuple[int, int]: pass

class LtNode(OpNode):
  def __floordiv__(self, b: Union[Node, int], _=False): return (self.a//b) < (self.b//b)
  def get_bounds(self) -> Tuple[int, int]:
    if isinstance(self.b, int):
      return (1, 1) if self.a.max < self.b else (0, 0) if self.a.min >= self.b else (0, 1)
    return (1, 1) if self.a.max < self.b.min else (0, 0) if self.a.min >= self.b.max else (0, 1)
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: return self.a.substitute(var_vals) < (self.b if isinstance(self.b, int) else self.b.substitute(var_vals))

class MulNode(OpNode):
  def __lt__(self, b: Union[Node, int]):
    if isinstance(b, Node) or isinstance(self.b, Node) or self.b == -1: return Node.__lt__(self, b)
    sgn = 1 if self.b > 0 else -1
    return Node.__lt__(self.a*sgn, (b + abs(self.b) - 1)//abs(self.b))
  def __mul__(self, b: Union[Node, int]): return self.a*(self.b*b) # two muls in one mul
  def __floordiv__(self, b: Union[Node, int], factoring_allowed=False): # NOTE: mod negative isn't handled right
    if self.b % b == 0: return self.a*(self.b//b)
    if b % self.b == 0 and self.b > 0: return self.a//(b//self.b)
    return Node.__floordiv__(self, b, factoring_allowed)
  def __mod__(self, b: Union[Node, int]):
    a = (self.a * (self.b%b))
    return Node.__mod__(a, b)
  def get_bounds(self) -> Tuple[int, int]:
    return (self.a.min*self.b, self.a.max*self.b) if self.b >= 0 else (self.a.max*self.b, self.a.min*self.b)
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: return self.a.substitute(var_vals) * (self.b if isinstance(self.b, int) else self.b.substitute(var_vals))

class DivNode(OpNode):
  def __floordiv__(self, b: Union[Node, int], _=False): return self.a//(self.b*b) # two divs is one div
  def get_bounds(self) -> Tuple[int, int]:
    assert self.a.min >= 0 and isinstance(self.b, int)
    return self.a.min//self.b, self.a.max//self.b
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: return self.a.substitute(var_vals) // self.b

class ModNode(OpNode):
  def __mod__(self, b: Union[Node, int]):
    if isinstance(b, Node) or isinstance(self.b, Node): return Node.__mod__(self, b)
    return self.a % b if gcd(self.b, b) == b else Node.__mod__(self, b)
  def __floordiv__(self, b: Union[Node, int], factoring_allowed=True):
    if (self.b % b == 0): return (self.a//b) % (self.b//b) # put the div inside mod
    return Node.__floordiv__(self, b, factoring_allowed)
  def get_bounds(self) -> Tuple[int, int]:
    assert self.a.min >= 0 and isinstance(self.b, int)
    return (0, self.b-1) if self.a.max - self.a.min >= self.b or (self.a.min != self.a.max and self.a.min%self.b >= self.a.max%self.b) else (self.a.min%self.b, self.a.max%self.b)
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: return self.a.substitute(var_vals) % self.b

class RedNode(Node):
  def __init__(self, nodes:List[Node]): self.nodes = nodes
  def vars(self): return functools.reduce(lambda l,x: l+x.vars(), self.nodes, [])

class SumNode(RedNode):
  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def __mul__(self, b: Union[Node, int]): return Node.sum([x*b for x in self.nodes]) # distribute mul into sum
  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def __floordiv__(self, b: Union[Node, int], factoring_allowed=True):
    fully_divided: List[Node] = []
    rest: List[Node] = []
    if isinstance(b, SumNode):
      nu_num = sum(node.b for node in self.flat_components if node.__class__ is NumNode)
      de_num = sum(node.b for node in b.flat_components if node.__class__ is NumNode)
      if nu_num > 0 and de_num and (d:=nu_num//de_num) > 0: return NumNode(d) + (self-b*d) // b
    if isinstance(b, Node):
      for x in self.flat_components:
        if x % b == 0: fully_divided.append(x // b)
        else: rest.append(x)
      if (sum_fully_divided:=create_rednode(SumNode, fully_divided)) != 0: return sum_fully_divided + create_rednode(SumNode, rest) // b
      return Node.__floordiv__(self, b, False)
    if b == 1: return self
    if not factoring_allowed: return Node.__floordiv__(self, b, factoring_allowed)
    fully_divided, rest = [], []
    _gcd = b
    divisor = 1
    for x in self.flat_components:
      if x.__class__ in (NumNode, MulNode):
        if x.b%b == 0: fully_divided.append(x//b)
        else:
          rest.append(x)
          _gcd = gcd(_gcd, x.b)
          if x.__class__ == MulNode and divisor == 1 and b%x.b == 0: divisor = x.b
      else:
        rest.append(x)
        _gcd = 1
    if _gcd > 1: return Node.sum(fully_divided) + Node.sum(rest).__floordiv__(_gcd) // (b//_gcd)
    if divisor > 1: return Node.sum(fully_divided) + Node.sum(rest).__floordiv__(divisor) // (b//divisor)
    return Node.sum(fully_divided) + Node.__floordiv__(Node.sum(rest), b)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def __mod__(self, b: Union[Node, int]):
    if isinstance(b, SumNode):
      nu_num = sum(node.b for node in self.flat_components if node.__class__ is NumNode)
      de_num = sum(node.b for node in b.flat_components if node.__class__ is NumNode)
      if nu_num > 0 and de_num and (d:=nu_num//de_num) > 0: return (self-b*d) % b
    if isinstance(b, Node) and (b - self).min > 0: return self # b - self simplifies the node
    new_nodes: List[Node] = []
    for x in self.nodes:
      if x.__class__ is NumNode: new_nodes.append(Variable.num(x.b%b))
      elif isinstance(x, MulNode): new_nodes.append(x.a * (x.b%b))
      else: new_nodes.append(x)
    return Node.__mod__(Node.sum(new_nodes), b)

  def __lt__(self, b:Union[Node,int]):
    lhs: Node = self
    if isinstance(b, int):
      new_sum = []
      for x in self.nodes:
        # TODO: should we just force the last one to always be the number
        if isinstance(x, NumNode): b -= x.b
        else: new_sum.append(x)
      lhs = Node.sum(new_sum)
      nodes = lhs.nodes if isinstance(lhs, SumNode) else [lhs]
      muls, others = partition(nodes, lambda x: isinstance(x, MulNode) and x.b > 0 and x.max >= b)
      if muls:
        # NOTE: gcd in python 3.8 takes exactly 2 args
        mul_gcd = b
        for x in muls: mul_gcd = gcd(mul_gcd, x.b)  # type: ignore  # mypy cannot tell x.b is int here
        all_others = Variable.sum(others)
        if all_others.min >= 0 and all_others.max < mul_gcd:
          lhs, b = Variable.sum([mul//mul_gcd for mul in muls]), b//mul_gcd
    return Node.__lt__(lhs, b)

  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: return Variable.sum([node.substitute(var_vals) for node in self.nodes])

  @property
  def flat_components(self): # recursively expand sumnode components
    new_nodes = []
    for x in self.nodes: new_nodes += (x.flat_components if isinstance(x, SumNode) else [x])
    return new_nodes

class AndNode(RedNode):
  def __floordiv__(self, b: Union[Node, int], _=True): return Variable.ands([x//b for x in self.nodes])
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node:
    subed = []
    for node in self.nodes:
      if not (sub:=node.substitute(var_vals)): return NumNode(0)
      subed.append(sub)
    return Variable.ands(subed)

def create_rednode(typ:Type[RedNode], nodes:List[Node]):
  ret = typ(nodes)
  if typ == SumNode: ret.min, ret.max = (sum([x.min for x in nodes]), sum([x.max for x in nodes]))
  elif typ == AndNode: ret.min, ret.max = (min([x.min for x in nodes]), max([x.max for x in nodes]))
  return create_node(ret)

@functools.lru_cache(maxsize=None)
def sym_rename(s) -> str: return f"s{sym_rename.cache_info().currsize}"
def sym_render(a: Union[Node, int], ops=None, ctx=None) -> str: return str(a) if isinstance(a, int) else a.render(ops, ctx)
def sym_infer(a: Union[Node, int], var_vals: Dict[Variable, int]) -> int:
  if isinstance(a, (int, float)): return a
  ret = a.substitute({k:Variable.num(v) for k, v in var_vals.items()})
  assert isinstance(ret, NumNode), f"sym_infer didn't produce NumNode from {a} with {var_vals}"
  return ret.b

# symbolic int
sint = Union[Node, int]
VariableOrNum = Union[Variable, NumNode]

render_python: Dict[Type, Callable] = {
  Variable: lambda self,ops,ctx: f"{self.expr}[{self.min}-{self.max}{'='+str(self.val) if self.val is not None else ''}]" if ctx == "DEBUG" else (f"Variable('{self.expr}', {self.min}, {self.max})" if ctx == "REPR" else f"{self.expr}"),
  NumNode: lambda self,ops,ctx: f"{self.b}",
  MulNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}*{sym_render(self.b,ops,ctx)})",
  DivNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}//{self.b})",
  ModNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}%{self.b})",
  LtNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}<{sym_render(self.b,ops,ctx)})",
  SumNode: lambda self,ops,ctx: f"({'+'.join(sorted([x.render(ops,ctx) for x in self.nodes]))})",
  AndNode: lambda self,ops,ctx: f"({' and '.join(sorted([x.render(ops,ctx) for x in self.nodes]))})"
}