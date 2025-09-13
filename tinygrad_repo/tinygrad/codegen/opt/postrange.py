from dataclasses import replace
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, KernelInfo
from tinygrad.helpers import colored
from tinygrad.codegen.opt.kernel import axis_colors

def rename_sink(s:UOp):
  if s.arg is not None and s.arg.name != "test": return None

  # get all ranges (sorted)
  rngs = sorted([u for u in s.parents if u.op is Ops.RANGE], key=lambda x: x.arg[0:-1])

  # add name to kernel
  name = "k" + colored('_', 'BLACK').join(['']+[colored(x.src[0].render(), axis_colors[x.arg[-1]]) for x in rngs])
  return s.replace(arg=KernelInfo(name=name) if s.arg is None else replace(s.arg, name=name))

pm_postrange_opt = PatternMatcher([
  (UPat(Ops.SINK, name="s"), rename_sink),
])
