import unittest
from dataclasses import dataclass, field
from tinygrad.uop.ops import PatternMatcher, UOp, graph_rewrite, Ops, UPat, GroupOp, RewriteNotReady

# we could insert CHILDREN node

@dataclass
class ChildrenContext:
  children: dict[UOp, list[UOp]]|None = None
  seen_children: dict[UOp, set[int]] = field(default_factory=dict)
  seen_consts:int = 0
  saved_seen_consts:int = 0

# this is a generic child labeller
def extract_children(ctx:ChildrenContext, x:UOp):
  if ctx.children is not None: return
  ctx.children = {k:list(v.keys()) for k,v in x.get_children_map().items() if len(v) > 1}

def mark_children(ctx:ChildrenContext, x:UOp):
  new_srcs = [(UOp(Ops.CHILD, s.dtype, src=(s,), arg=(ctx.children[s].index(x), len(ctx.children[s]))) if s in ctx.children else s) for s in x.src]
  return x.replace(src=tuple(new_srcs))

pm_children = PatternMatcher([
  (UPat(Ops.SINK, name="x"), extract_children),
  (UPat(GroupOp.All-{Ops.CHILD}, name="x"), mark_children),
])

# this is a generic pattern
def visit_child(ctx:ChildrenContext, x:UOp):
  if x.src[0] not in ctx.seen_children: ctx.seen_children[x.src[0]] = set()
  ctx.seen_children[x.src[0]].add(x.arg[0])
  if len(ctx.seen_children[x.src[0]]) != x.arg[1]:
    print(f"visit CHILD {x.arg} bottom up -- not ready {ctx.seen_children[x.src[0]]}")
    raise RewriteNotReady
  print(f"visit CHILD {x.arg} bottom up -- READY {ctx.seen_children[x.src[0]]}")

pm_child_visitor = PatternMatcher([
  (UPat(Ops.CHILD, name="x"), visit_child),
])

# this is for the test
def see_const(ctx:ChildrenContext, c:UOp): ctx.seen_consts += c.arg
def save_seen_consts(ctx:ChildrenContext, x:UOp): ctx.saved_seen_consts = ctx.seen_consts
pm_consts = PatternMatcher([
  (UPat(Ops.DEFINE_VAR, name="x"), save_seen_consts),
  (UPat()+UPat.cvar("c"), see_const),
])

class TestChildrenRewrite(unittest.TestCase):
  def test_not_ready(self):
    a = UOp.variable("a", 0, 10).exp2()
    b = a+2
    c = a+3
    d = b+c
    sink = d.sink()

    # without children and not ready, we don't see both adds before the DEFINE_VAR
    ctx = ChildrenContext()
    sink = graph_rewrite(sink, pm_consts, ctx=ctx, bottom_up=True)
    self.assertNotEqual(ctx.seen_consts, ctx.saved_seen_consts)

    # with children and not ready we do
    ctx = ChildrenContext()
    sink = graph_rewrite(sink, pm_children, ctx=ctx, bottom_up=True)
    sink = graph_rewrite(sink, pm_child_visitor+pm_consts, ctx=ctx, bottom_up=True)
    self.assertEqual(ctx.seen_consts, ctx.saved_seen_consts)

if __name__ == '__main__':
  unittest.main()
