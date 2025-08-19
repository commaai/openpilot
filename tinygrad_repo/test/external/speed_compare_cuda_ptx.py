import itertools
from tinygrad import Device
from tinygrad.engine.realize import CompiledRunner, get_program
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.helpers import getenv, colorize_float
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.codegen.opt.search import bufs_from_lin
from tinygrad.runtime.ops_cuda import PTXCompiler, PTXRenderer, CUDACompiler

if __name__ == "__main__":
  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
  # no bfloat16 for ptx at the moment
  ast_strs = [x for x in ast_strs if "dtypes.bfloat16" not in x]
  dev = Device["CUDA"]
  ptx = PTXRenderer(dev.arch)

  # NUM=112 python3 test/external/speed_compare_cuda_ptx.py

  single = getenv("NUM", -1)
  if single != -1: ast_strs = ast_strs[single:single+1]

  average_tm_cuda, average_tm_ptx = 0, 0
  for num,ast in enumerate(ast_strs):
    # cuda compile
    dev.compiler = CUDACompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=dev.renderer)
    lin.apply_opts(hand_coded_optimizations(lin))
    cuda_prg = CompiledRunner(get_program(lin.get_optimized_ast(), lin.opts))

    bufs = bufs_from_lin(lin)

    # ptx compile
    dev.compiler = PTXCompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=ptx)
    lin.apply_opts(hand_coded_optimizations(lin))
    ptx_prg = CompiledRunner(get_program(lin.get_optimized_ast(), lin.opts))

    # warmup
    try:
      cuda_prg(bufs, {}, wait=True)
    except RuntimeError:
      print("cuda failed ast:", num)
      continue
    ptx_prg(bufs, {}, wait=True)

    tm_cuda, tm_ptx = [], []
    for i in range(5):
      tm_cuda.append(cuda_prg(bufs, {}, wait=True))
      tm_ptx.append(ptx_prg(bufs, {}, wait=True))
    average_tm_cuda += min(tm_cuda)
    average_tm_ptx += min(tm_ptx)
    ratio = min(tm_ptx)/min(tm_cuda)
    print(f"{average_tm_ptx/average_tm_cuda:5.2f}x -- {num:4d} {colorize_float(ratio)}  {min(tm_ptx)*1e6:7.2f} us", lin.name)
    if ratio > 1.5:
      def fix(x): return x.replace('\t', ' ').strip()
      ll1, ll2 = cuda_prg.lib.decode().split('\n'), ptx_prg.lib.decode().split('\n')
      if single != -1:
        for ln, (l1, l2) in enumerate(itertools.zip_longest(ll1, ll2, fillvalue='')):
          print(f"{ln:5d} | {fix(l1):80s} | {fix(l2):80s}")
      print(len(ll1), len(ll2), "RATIO", ratio, "us", min(tm_ptx)*1e6)
