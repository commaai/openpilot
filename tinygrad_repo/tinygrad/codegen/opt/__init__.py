# opt opinionatedly transforms an ast into an optimized ast using either heuristics or beam search
from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from tinygrad.uop.ops import AxisType

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

axis_letters = {AxisType.GLOBAL: "g", AxisType.THREAD: "t", AxisType.LOCAL: "l", AxisType.WARP: "w", AxisType.LOOP: "L", AxisType.UPCAST: "u",
                AxisType.GROUP_REDUCE: "G", AxisType.REDUCE: "R", AxisType.UNROLL: "r"}
axis_colors = {AxisType.GLOBAL: "blue", AxisType.THREAD: "BLUE", AxisType.LOCAL: "cyan", AxisType.WARP: "CYAN", AxisType.LOOP: "WHITE",
               AxisType.UPCAST: "yellow", AxisType.GROUP_REDUCE: "green", AxisType.REDUCE: "red", AxisType.UNROLL: "magenta"}

class KernelOptError(Exception): pass
def check(cond:bool, msg:str=""):
  if not cond: raise KernelOptError(msg)
