# opt opinionatedly transforms an ast into an optimized ast using either heuristics or beam search

from tinygrad.codegen.opt.kernel import Kernel
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.uop.ops import UOp, PatternMatcher, UPat, Ops, KernelInfo
from tinygrad.helpers import NOOPT, BEAM, USE_TC, getenv
from tinygrad.renderer import Renderer
from tinygrad.uop.spec import type_verify

def get_optimized_ast(ast:UOp, renderer:Renderer) -> UOp:
  """
  Optimize an AST based on heuristics or BEAM search.

  Args:
    ast: The Ops.SINK rooted AST
    renderer: The renderer used to generate the code

  Returns:
    The Ops.SINK rooted AST transformed to apply the opts and with a KernelInfo in the arg.
  """

  assert ast.arg is None, "no opt if there's an arg"
  k = Kernel(ast, opts=renderer)
  if not NOOPT:
    if not k.apply_tensor_cores(USE_TC.value): k.apply_opts(hand_coded_optimizations(k))
    if BEAM >= 1:
      from tinygrad.codegen.opt.search import beam_search, bufs_from_lin
      kb = Kernel(ast, opts=renderer)
      rawbufs = bufs_from_lin(kb, allocate=False)
      k = beam_search(kb, rawbufs, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))
  return ast.replace(arg=KernelInfo(opts_to_apply=tuple(k.applied_opts)))

pm_get_optimization = PatternMatcher([
  (UPat(Ops.SINK, name="ast"), lambda ctx,ast: get_optimized_ast(ast, ctx) if ast.arg is None and ast.src[0].st is not None else None),
])

def apply_opt(ast:UOp, renderer:Renderer):
  k = Kernel(ast, opts=renderer)
  k.apply_opts(ast.arg.opts_to_apply)
  ret = k.get_optimized_ast()
  if __debug__: type_verify(list(ret.toposort()))
  return ret

pm_do_optimize = PatternMatcher([
  (UPat(Ops.SINK, name="ast"), lambda ctx,ast: apply_opt(ast, ctx) if ast.arg is not None and ast.arg.opts_to_apply is not None else None),
])
