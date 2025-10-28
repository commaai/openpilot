from tinygrad.uop.ops import Ops, UOp, resolve, can_pad, GroupOp, UPat, PatternMatcher, graph_rewrite
from tinygrad.helpers import all_int, prod, unwrap, dedup, DONT_REALIZE_EXPAND, DONT_GROUP_REDUCES, FUSE_CONV_BW
from tinygrad.shape.shapetracker import ShapeTracker

ALWAYS_CONTIGUOUS: set[Ops] = {Ops.CONTIGUOUS, Ops.ASSIGN, Ops.COPY, Ops.BUFFER, Ops.BUFFER_VIEW,
                     Ops.CONST, Ops.BIND, Ops.DEVICE, Ops.MSELECT, Ops.MSTACK, Ops.DEFINE_GLOBAL,
                     Ops.DEFINE_LOCAL, Ops.DEFINE_REG, Ops.LOAD}

# **** Grouper decides which of the UOps realize

def realize(ctx:dict[UOp, None], tr:UOp) -> None: ctx[tr] = None

def realize_parents(ctx:dict[UOp, None], rb:UOp) -> None:
  for s in rb.src:
    if s.op not in ALWAYS_CONTIGUOUS: ctx[s] = None

def realize_before_view(ctx:dict[UOp, None], view:UOp, tr:UOp) -> None:
  st = unwrap(view.st)
  # always realize unsafe pad ops before masked view
  if any(v.mask is not None for v in st.views) and not can_pad(tr, ctx): return realize(ctx, tr)
  # fold simple pads
  if len(st.views) == 1 and (m:=st.views[-1].mask) is not None and all_int(tr.shape) and resolve(prod(tr.shape) >= prod([y-x for x,y in m])): return
  # realize before expand
  if resolve(prod(tr.shape) < prod(st.shape)) and not DONT_REALIZE_EXPAND: return realize(ctx, tr)

do_realize = PatternMatcher([
  # always realize SINK parents
  (UPat(Ops.SINK, name="s"), lambda ctx,s: ctx.update((x.base, None) for x in s.src if x.base.op not in ALWAYS_CONTIGUOUS)),
  # always realize ASSIGN/CONTIGUOUS/COPY/BUFFER_VIEW
  (UPat({Ops.ASSIGN, Ops.CONTIGUOUS, Ops.COPY, Ops.BUFFER_VIEW}, name="tr"), realize),
  # realize before expand or unsafe pad ops
  (UPat(Ops.VIEW, src=(UPat(GroupOp.All-ALWAYS_CONTIGUOUS, name="tr"),), name="view"), realize_before_view),
  # realize parents of COPY, MSELECT, MSTACK
  (UPat((Ops.COPY, Ops.MSELECT, Ops.MSTACK), name="rb"), realize_parents),
])

def recursive_group(tr:UOp, st:ShapeTracker, r:UOp, children:dict[UOp, dict[UOp, None]], realizes:dict[UOp, None],
                    reduce_for_op:dict[UOp, UOp], group:dict[UOp, None], cache:dict[tuple[UOp, ShapeTracker], None]) -> None:
  if (tr, st) in cache: return
  cache.setdefault((tr, st))
  rsize = unwrap(r.st).size
  if tr in realizes and tr is not r:
    # can only fuse contiguous
    # max one reduceop per kernel
    if not st.contiguous or st.size != rsize or tr in reduce_for_op: group.setdefault(r)
    return group.setdefault(tr)
  for tr_next in children.get(tr, {}):
    # max one reduceop per kernel
    if tr_next.op is Ops.REDUCE_AXIS: return group.setdefault(r)
    # can only fuse contiguous
    if len(st_childs:=dedup(unwrap(x.st) for x in tr_next.src if x.base == tr)) > 1: return group.setdefault(r)
    recursive_group(tr_next, st+st_childs[0], r, children, realizes, reduce_for_op, group, cache)

def group_realizes(sink:UOp) -> dict[UOp, None]:
  # start by adding uops that always realize
  realizes: dict[UOp, None] = {}
  sink = graph_rewrite(sink, do_realize, ctx=realizes, name="do_realize")
  if DONT_GROUP_REDUCES: return realizes

  # construct children graph (only for bases)
  children: dict[UOp, dict[UOp, None]] = {}
  assigns: dict[UOp, None] = {}
  for u in (toposort:=sink.toposort()):
    if u.op in {Ops.VIEW, Ops.SINK}: continue
    if u.op is Ops.ASSIGN: assigns[u.buf_uop] = None
    for s in u.src: children.setdefault(s.base, {})[u] = None

  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: dict[UOp, UOp] = {}
  double_reduces: list[UOp] = []
  for r in toposort:
    if r.op is not Ops.REDUCE_AXIS: continue
    if len(r.arg) == 3 and r.arg[2] is True: continue
    if FUSE_CONV_BW and r.src[0].base.op is Ops.REDUCE_AXIS and r.src[0] is not r.src[0].base: double_reduces.append(r)
    if r in realizes: continue
    group: dict[UOp, None] = {}
    recursive_group(r, unwrap(r.st), r, children, realizes, reduce_for_op, group, cache={})
    # max one reduceop per kernel
    can_chase = all(tr not in reduce_for_op for tr in group)
    for u in r.toposort(gate=lambda u: u not in realizes):
      if u.op is Ops.REDUCE_AXIS and u.src[0].base.op is Ops.CONST:
        can_chase = False
        break
    # TODO: forced_realize exists because the scheduler is incapable of checking for self-contained DAGs
    forced_realize = r in group
    # can only have one output
    if not forced_realize and len(group) > 1: forced_realize = True
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and (assign_targets:={x.buf_uop for x in group if x.op is Ops.ASSIGN}):
      parents = [r, *group]
      while parents and not forced_realize:
        p = parents.pop().base
        if p.op is Ops.BUFFER and p in assigns and p not in assign_targets: forced_realize, can_chase = True, False
        if p in realizes: continue
        parents.extend(p.src)
    if forced_realize or not group:
      tr = r
      if can_chase:
        # can chase this down to contiguous children
        st = unwrap(tr.st)
        while len(lst:=children.get(tr, {})) == 1:
          tr_next = next(iter(lst))
          st_childs = dedup(unwrap(s.st) for s in tr_next.src if s.base is tr)
          if len(st_childs) > 1: break
          if st.size != st_childs[0].size: break
          st = st + st_childs[0]
          if not st.contiguous or tr_next.op is Ops.REDUCE_AXIS: break
          tr = tr_next
        # don't cast to higher size before store (tr cannot be realized if forced_realize)
        if tr.op is Ops.CAST and tr.dtype.itemsize > tr.src[0].dtype.itemsize:
          tr = tr.src[0].base
      group = {tr: None}
      realizes[tr] = None
    reduce_for_op.update((tr, r) for tr in group)
  # fuse double reduces with no other child
  for reduceop in double_reduces:
    top_reduce = reduceop.src[0].base
    if len(children.get(top_reduce, {})) == 1: del realizes[top_reduce]
  return realizes
