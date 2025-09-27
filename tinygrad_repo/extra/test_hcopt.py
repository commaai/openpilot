import time
from extra.optimization.helpers import load_worlds, ast_str_to_ast
from tinygrad import Device
from tinygrad.codegen.lowerer import pm_lowerer, get_index
from tinygrad.uop.ops import graph_rewrite
from tinygrad.codegen.opt.kernel import Kernel
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.helpers import getenv

if __name__ == "__main__":
  renderer = Device.default.renderer
  ast_strs = load_worlds()
  if (n:=getenv("N", -1)) != -1: ast_strs = ast_strs[n:n+1]
  good = 0
  for i, ast_str in enumerate(ast_strs):
    ast = ast_str_to_ast(ast_str)

    st = time.perf_counter()
    lin = Kernel(ast, renderer)
    opt1 = hand_coded_optimizations(lin)
    et_lin = time.perf_counter() - st

    lowered = graph_rewrite(ast, pm_lowerer, ctx=get_index(ast), bottom_up=True)
    st = time.perf_counter()
    sch = Scheduler(lowered, renderer)
    sch.convert_loop_to_global()
    sch.simplify_merge_adjacent()
    opt2 = hand_coded_optimizations(sch)
    et_sch = time.perf_counter() - st

    if opt1 != opt2:
      print(f"******* {i:6d}")
      print("Kernel:    ", lin.colored_shape(), "->", lin.apply_opts(opt1).colored_shape())
      print("Scheduler: ", sch.colored_shape(), "->", sch.apply_opts(opt2).colored_shape())
      print(opt1)
      print(opt2)
    else:
      good += 1
      print(f"******* {i:6d} MATCH {good/(i+1)*100:.2f}% -- {et_lin/et_sch:4.2f}x speedup")
