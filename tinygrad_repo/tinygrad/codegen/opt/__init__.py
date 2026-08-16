# opt opinionatedly transforms an ast into an optimized ast using either heuristics or beam search
from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass

class OptOps(Enum):
  TC = auto(); UPCAST = auto(); UNROLL = auto(); LOCAL = auto(); THREAD = auto() # noqa: E702
  GROUP = auto(); GROUPTOP = auto(); NOLOCALS = auto(); PADTO = auto(); SWAP = auto() # noqa: E702
  def __lt__(self, x:OptOps): return self.value < x.value

@dataclass(frozen=True, order=True)
class Opt:
  op: OptOps
  axis: int|None = None
  arg: int|tuple|None = None
  def __repr__(self): return f"Opt(op={self.op}, axis={self.axis}, arg={self.arg})"

class KernelOptError(Exception): pass
def check(cond:bool, msg:str=""):
  if not cond: raise KernelOptError(msg)
