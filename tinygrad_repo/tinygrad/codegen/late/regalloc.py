import itertools
from tinygrad.helpers import dedup
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.isa import ISARenderer, Register, rdef, LinearContext
from typing import Any

PSEUDO_OPS = {Ops.CONST, Ops.CAST, Ops.BITCAST, Ops.NOOP, Ops.AFTER, Ops.BARRIER, Ops.GROUP, Ops.STACK}

class LinearScanRegallocContext:
  # returns the uop that defines the virtual register
  def vdef(self, v:Register) -> UOp: return self.uops[self.live_range[v][0]]
  def __init__(self, ctx:LinearContext, uops:list[UOp], ren:ISARenderer):
    self.uops = uops
    self.ren = ren
    self.idx = itertools.count()

    # compute live ranges
    self.live_range: dict[Register, list[int]] = {}
    lr = self.live_range
    loops: dict[int, int] = {} # the interval of each loop, from its RANGE to the last uop that reads that RANGE
    for idx,u in reversed(list(enumerate(uops))):
      if u.op in PSEUDO_OPS: continue
      defs = u.tag if isinstance(u.tag, tuple) else ()
      for v in defs + tuple(rdef(s) for s in dedup(u.src)):
        if isinstance(v, Register): lr.setdefault(v, []).insert(0, idx)
      for v in defs:
        if v in lr and (n:=max((e for s,e in loops.items() if s <= lr[v][-1] < e), default=None)): lr[v].append(n)
      if u.op is Ops.RANGE: loops[idx] = max(j for j,x in enumerate(uops) if u in x.src)

    # allocate registers
    self.locals: dict[UOp, UOp] = {}
    self.spills: dict[Register, Any] = {} # mapping from virtual to arbitrary spill slot
    self.reals: dict[int, dict[Register, Register]] = {} # mapping from virtual to real at each program point
    self.insert_before: dict[int, list[tuple[Register, Register]]] = {} # fills to be inserted at each program point
    live: dict[Register, Register] = {} # mapping from virtual to real that's currently assigned to it
    live_ins: list[dict[Register, Register]] = [] # mapping from virtual to real at loop entry

    def alloc(cons:tuple[Register, ...], i:int) -> Register:
      live_inv = {v:k for k,v in live.items()}
      # allocate the best register. Registers not in live or not used again are free and have priority,
      # otherwise pick the one with the furthest next use. Regs that appear first in cons have priority in case of a tie
      reg,vreg = max(((r,live_inv.get(r)) for r in cons),
                    key=lambda rv: next((j-i for j in ([] if rv[1] is None else lr[rv[1]]) if j >= i), len(uops)))
      return live.pop(vreg) if vreg is not None else reg

    # assign register to spilled virtual and record load to be emitted before current uop, also assign it a stack slot
    def fill(v:Register, i:int, cons:tuple[Register, ...]|None=None) -> Register:
      if v not in self.spills:
        self.spills[v] = ctx.assign_spill_slot(v, self.vdef(v))
      r = alloc(cons if cons is not None else v.cons, i)
      self.insert_before.setdefault(i, []).append((v, r))
      return r

    for i,u in enumerate(uops):
      if u.op in PSEUDO_OPS: continue
      # allocate uses
      for s in u.src:
        # HACK: cause of later hacks to lower range
        if u.op is Ops.END: continue
        if not isinstance(v:=rdef(s), Register): continue
        if v not in live: live[v] = fill(v, i)
        self.reals.setdefault(i, {})[v] = live[v]

      # allocate defs
      if isinstance(u.tag, tuple):
        for j,v in enumerate(u.tag):
          # register should only be defined once
          assert isinstance(v, Register) and lr[v][0] == i
          cons = v.cons
          # two address instructions (src is reused by def) can only coalesce reused src. reused src goes first to get priority in case of a tiebreak
          if ren.is_two_address(u) and j == 0:
            uses = tuple(live.get(rdef(s)) for s in u.src)
            cons = ((uses[0],) if uses[0] in cons else ()) + tuple(r for r in cons if r not in uses)
          # HACK: cause the range is missing the comparison
          live[v] = alloc(cons, i+1 if u.op is not Ops.RANGE else i)
          self.reals.setdefault(i, {})[v] = live[v]

      # loop prologue, avoid loading inside the loop
      if u.op is Ops.RANGE:
        # we move to registers vars used in the loop sorted by next use, vars not used in the loop will not be reloaded in the epilogue
        used_in_loop = [v for v in live.keys() | self.spills.keys() if any(i <= l < loops[i] for l in lr[v])]
        sorted_uses = sorted(used_in_loop, key=lambda k: (next(l-i for l in lr[k] if l >= i), lr[k][0], k.name, k.index))
        live_in: dict[Register, Register] = {}
        for v in sorted_uses:
          # if all the possible registers are already in live_in there's no space for this var
          if set(v.cons).issubset(live_in.values()): continue
          if v not in live: live[v] = fill(v, i)
          live_in[v] = live[v]
        live_ins.append(live_in)

      # loop epilogue, reload registers that were live at loop entry
      if u.op is Ops.END:
        # TODO: if a uop is in a different reg in live out vs live in move between registers instead of loading
        # TODO: don't reload if first use in loop is a load
        for v,r in live_ins.pop().items():
          if v not in live or live[v] != r: live[v] = fill(v, i, (r,))

def regalloc_rewrite(ctx:LinearScanRegallocContext, x:UOp):
  i = next(ctx.idx)
  if x.op in PSEUDO_OPS: return None
  nsrc = []
  for j,s in enumerate(x.src):
    # v here is the virtual defined by the original s as s is the rewritten version
    if i in ctx.reals and (v:=rdef(ctx.uops[i].src[j])) in ctx.spills: nsrc.append(ctx.ren.fill(ctx.spills[v], ctx.vdef(v), ctx.reals[i][v]))
    else: nsrc.append(s)
  ndefs = tuple(ctx.reals[i][v] for v in x.tag) if isinstance(x.tag, tuple) else x.tag
  nx = x.replace(src=tuple(nsrc), tag=ndefs)

  before = [ctx.ren.fill(ctx.spills[v], ctx.vdef(v), r) for v,r in ctx.insert_before.get(i, [])]
  after = [ctx.ren.spill(ctx.spills[v], nx) for v in x.tag if v in ctx.spills] if isinstance(x.tag, tuple) else []

  return nx, before + [nx] + after

pm_regalloc_rewrite = PatternMatcher([
  (UPat({Ops.INS, Ops.RANGE, Ops.END, Ops.BUFFER, Ops.PARAM, Ops.SPECIAL} | PSEUDO_OPS, name="x"), regalloc_rewrite),
])
