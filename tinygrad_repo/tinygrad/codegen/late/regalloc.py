import itertools
from tinygrad.helpers import dedup, get_single_element
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.x86 import ISARenderer, vdef, register_choices, LinearContext

# ops that participate in register allocation
REGALLOC_OPS = {Ops.CALL, Ops.BUFFER, Ops.RANGE, Ops.END}

class LinearScanRegallocContext:
  # returns the uop that defines the virtual register
  def vdef(self, v:UOp) -> UOp: return self.uops[self.live_range[v][0]]
  def __init__(self, ctx:LinearContext, uops:list[UOp], ren:ISARenderer):
    self.uops = uops = [u for u in uops if u.op in REGALLOC_OPS]
    self.ren = ren
    self.idx = itertools.count()

    # compute live ranges
    self.live_range: dict[UOp, list[int]] = {}
    self.defs = [register_choices(u) for u in uops]
    choices = {p:rs for defs in self.defs for p,rs in defs.items()}
    lr = self.live_range
    loops: dict[int, int] = {} # the interval of each loop, from its RANGE to the last uop that reads that RANGE
    for idx,u in reversed(list(enumerate(uops))):
      defs = tuple(self.defs[idx])
      for v in defs + tuple(vdef(s) for s in dedup(u.src)):
        if v is not None: lr.setdefault(v, []).insert(0, idx)
      for v in defs:
        if (n:=max((e for s,e in loops.items() if s <= lr[v][-1] < e), default=None)): lr[v].append(n)
      if u.op is Ops.RANGE: loops[idx] = max(j for j,x in enumerate(uops) if u in x.src)

    # allocate registers
    self.spills: dict[UOp, int] = {} # mapping from virtual to stack offset
    self.reals: dict[int, dict[UOp, UOp]] = {} # mapping from virtual to real at each program point
    self.insert_before: dict[int, list[tuple[UOp, UOp]]] = {} # fills to be inserted at each program point
    live: dict[UOp, UOp] = {} # mapping from virtual to real that's currently assigned to it
    live_ins: list[dict[UOp, UOp]] = [] # mapping from virtual to real at loop entry

    def alloc(cons:tuple[UOp, ...], i:int) -> UOp:
      live_inv = {v:k for k,v in live.items()}
      # allocate the best register. Registers not in live or not used again are free and have priority,
      # otherwise pick the one with the furthest next use. Regs that appear first in cons have priority in case of a tie
      reg,vreg = max(((r,live_inv.get(r)) for r in cons),
                    key=lambda rv: next((j-i for j in ([] if rv[1] is None else lr[rv[1]]) if j >= i), len(uops)))
      return live.pop(vreg) if vreg is not None else reg

    # assign register to spilled virtual and record load to be emitted before current uop, also assign it a stack slot
    def fill(v:UOp, i:int, cons:tuple[UOp, ...]|None=None) -> UOp:
      r = alloc(cons if cons is not None else choices[v], i)
      if v not in self.spills: self.spills[v] = ctx.assign_spill_slot(r)
      self.insert_before.setdefault(i, []).append((v, r))
      return r

    for i,u in enumerate(uops):
      # allocate uses
      for s in u.src:
        if (v:=vdef(s)) is None: continue
        if v not in live: live[v] = fill(v, i)
        self.reals.setdefault(i, {})[v] = live[v]

      # allocate defs
      if self.defs[i]:
        for j,v in enumerate(self.defs[i]):
          if v not in live:
            cons = choices[v]
            # two address instructions (src is reused by def) can only coalesce reused src.
            # reused src goes first to get priority in case of a tiebreak
            if ren.is_two_address(u) and j == 0:
              uses = tuple(live.get(src) if (src:=vdef(s)) is not None else None for s in u.src_without_body)
              cons = ((uses[0],) if uses[0] in cons else ()) + tuple(r for r in cons if r not in uses)
            live[v] = alloc(cons, i+1)
          self.reals.setdefault(i, {})[v] = live[v]

      # loop prologue, avoid loading inside the loop
      if u.op is Ops.RANGE:
        # we move to registers vars used in the loop sorted by next use, vars not used in the loop will not be reloaded in the epilogue
        used_in_loop = [v for v in live.keys() | self.spills.keys() if any(i <= l < loops[i] for l in lr[v])]
        sorted_uses = sorted(used_in_loop, key=lambda k: (next(l-i for l in lr[k] if l >= i), lr[k][0], k.key))
        live_in: dict[UOp, UOp] = {}
        for v in sorted_uses:
          # if all the possible registers are already in live_in there's no space for this var
          if set(choices[v]).issubset(live_in.values()): continue
          if v not in live: live[v] = fill(v, i)
          live_in[v] = live[v]
        live_ins.append(live_in)

      # loop epilogue at the loop's last reader, reload registers that were live at loop entry
      if i in loops.values():
        # TODO: if a uop is in a different reg in live out vs live in move between registers instead of loading
        # TODO: don't reload if first use in loop is a load
        live_in = live_ins.pop()
        for v,r in live_in.items():
          if live.get(v) is not r: live[v] = fill(v, i, (r,))
        # a spilled var that only got a register inside the loop must be forced to reload from spill slot
        for v in list(live.keys()):
          if v in self.spills and v not in live_in: live.pop(v)

def regalloc_rewrite(ctx:LinearScanRegallocContext, x:UOp):
  i = next(ctx.idx)
  nsrc = []
  for j,s in enumerate(x.src):
    # v here is the virtual defined by the original s as s is the rewritten version
    if (v:=vdef(ctx.uops[i].src[j])) in ctx.reals.get(i, {}): nsrc.append(ctx.reals[i][v].cast(s.dtype))
    else: nsrc.append(s)
  nx = x.replace(src=tuple(nsrc))
  if (defs:=ctx.defs[i]):
    r = tuple(ctx.reals[i][v] for v in defs)
    nx = get_single_element(r).cast(nx.dtype) if nx.op is Ops.BUFFER else UOp.ins(nx.body.arg.code, nx.src_without_body, nx.dtype, *r)

  before = [ctx.ren.fill(ctx.spills[v], ctx.vdef(v), r) for v,r in ctx.insert_before.get(i, [])]
  after = [ctx.ren.spill(ctx.spills[v], nx) for v in defs if v in ctx.spills]

  return nx, before + [nx] + after

pm_regalloc_rewrite = PatternMatcher([
  (UPat(REGALLOC_OPS, name="x"), regalloc_rewrite),
])
