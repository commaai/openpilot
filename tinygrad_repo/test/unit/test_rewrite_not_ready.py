import unittest
from dataclasses import dataclass, field
from tinygrad.uop.ops import PatternMatcher, UOp, graph_rewrite, Ops, UPat, GroupOp, RewriteNotReady

# we could insert CHILDREN node

@dataclass
class ChildrenContext:
  children: dict[UOp, list[UOp]]|None = None

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

@dataclass
class TestContext:
  seen_children: dict[UOp, set[int]] = field(default_factory=dict)
  ready_children: dict[UOp, set[int]] = field(default_factory=dict)
  seen_consts:int = 0
  saved_seen_consts:int = 0
  exp2_visit_count:int = 0

# this is a generic pattern
def visit_child(ctx:ChildrenContext, x:UOp):
  if x.src[0] not in ctx.seen_children:
    ctx.seen_children[x.src[0]] = set()
    ctx.ready_children[x.src[0]] = set()
  ctx.seen_children[x.src[0]].add(x.arg[0])
  if len(ctx.seen_children[x.src[0]]) != x.arg[1]:
    print(f"visit CHILD {x.arg} bottom up -- not ready {ctx.seen_children[x.src[0]]}")
    raise RewriteNotReady
  print(f"visit CHILD {x.arg} bottom up -- READY {ctx.seen_children[x.src[0]]}")
  ctx.ready_children[x.src[0]].add(x.arg[0])

pm_child_visitor = PatternMatcher([
  (UPat(Ops.CHILD, name="x"), visit_child),
])

# this is for the test
def see_const(ctx:ChildrenContext, c:UOp): ctx.seen_consts += c.arg
def see_exp2(ctx:ChildrenContext): ctx.exp2_visit_count += 1
def save_seen_consts(ctx:ChildrenContext, x:UOp): ctx.saved_seen_consts = ctx.seen_consts
pm_consts = PatternMatcher([
  (UPat(Ops.DEFINE_VAR, name="x"), save_seen_consts),
  (UPat()+UPat.cvar("c"), see_const),
  (UPat(Ops.EXP2), see_exp2),
])

class TestChildrenRewrite(unittest.TestCase):
  def test_not_ready_double_simple(self):
    global_a = UOp.variable("a", 0, 10).exp2()
    inter = (global_a+global_a).exp2()
    global_sink = (inter+inter).sink()

    sink = graph_rewrite(global_sink, pm_children, ctx=ChildrenContext(), bottom_up=True)
    ctx = TestContext()
    graph_rewrite(sink, pm_consts, ctx=ctx, bottom_up=True)
    self.assertEqual(ctx.exp2_visit_count, 2)

  def test_not_ready_double(self):
    global_a = UOp.variable("a", 0, 10).exp2()
    inter = ((global_a+1000)+(global_a+100)).exp2()
    global_sink = ((inter+10)+(inter+1)).sink()

    sink = graph_rewrite(global_sink, pm_children, ctx=ChildrenContext(), bottom_up=True)
    print("test_not_ready_double")
    ctx = TestContext()
    graph_rewrite(sink, pm_child_visitor+pm_consts, ctx=ctx, bottom_up=True)
    self.assertEqual(ctx.exp2_visit_count, 2)
    self.assertEqual(ctx.seen_consts, ctx.saved_seen_consts)
    self.assertEqual(ctx.seen_consts, 1111)

  def test_in_srcs_twice(self):
    global_a = UOp.variable("a", 0, 10).exp2()
    global_sink = (global_a+global_a).sink()

    ctx = TestContext()
    graph_rewrite(global_sink, pm_consts, ctx=ctx, bottom_up=True)
    self.assertEqual(ctx.exp2_visit_count, 1)

  def test_not_ready(self):
    global_a = UOp.variable("a", 0, 10).exp2()
    global_sink = ((global_a+2)+(global_a+3)).sink()

    # without children and not ready, we don't see both adds before the DEFINE_VAR
    ctx = TestContext()
    graph_rewrite(global_sink, pm_consts, ctx=ctx, bottom_up=True)
    self.assertNotEqual(ctx.seen_consts, ctx.saved_seen_consts)
    self.assertEqual(ctx.exp2_visit_count, 1)

    # with children and not ready we do
    sink = graph_rewrite(global_sink, pm_children, ctx=ChildrenContext(), bottom_up=True)
    ctx = TestContext()
    graph_rewrite(sink, pm_child_visitor+pm_consts, ctx=ctx, bottom_up=True)
    self.assertEqual(ctx.seen_consts, ctx.saved_seen_consts)
    self.assertEqual(ctx.exp2_visit_count, 1)
    self.assertSetEqual(list(ctx.ready_children.values())[0], {0,1})

if __name__ == '__main__':
  unittest.main()
