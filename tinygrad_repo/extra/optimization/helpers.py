# stuff needed to unpack a kernel
from tinygrad import Variable
from tinygrad.codegen.kernel import Opt, OptOps
from tinygrad.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
inf, nan = float('inf'), float('nan')
UOps = Ops

# kernel unpacker
from tinygrad.codegen.kernel import Kernel
def ast_str_to_ast(ast_str:str) -> UOp: return eval(ast_str)
def ast_str_to_lin(ast_str:str, opts=None): return Kernel(ast_str_to_ast(ast_str), opts=opts)
def kern_str_to_lin(kern_str:str, opts=None):
  (ast, applied_opts,) = eval(kern_str)
  k = Kernel(ast, opts=opts)
  k.apply_opts(applied_opts)
  return k

# load worlds, a dataset of about 12k kernels
import gzip
from pathlib import Path
import random
from tinygrad.helpers import dedup, DEBUG
def load_worlds(filter_reduce=True, filter_noimage=True, filter_novariable=True):
  fn = Path(__file__).parent.parent / "datasets/sops.gz"
  ast_strs = dedup(gzip.open(fn).read().decode('utf-8').strip().split("\n"))
  assert len(ast_strs) > 5000, f"dataset size = {len(ast_strs)} is too small"
  if DEBUG >= 1: print(f"loaded {len(ast_strs)=} before filters")
  if filter_reduce: ast_strs = [x for x in ast_strs if "REDUCE_AXIS" in x]
  if filter_noimage: ast_strs = [x for x in ast_strs if "dtypes.image" not in x]
  if filter_novariable: ast_strs = [x for x in ast_strs if "DEFINE_VAR" not in x]
  if DEBUG >= 1: print(f"loaded {len(ast_strs)=} after filters {filter_reduce=}, {filter_noimage=}, {filter_novariable=}")
  random.seed(1337)
  random.shuffle(ast_strs)
  return ast_strs

def assert_same_lin(l1, l2):
  assert l1.colored_shape() == l2.colored_shape()
  assert all(x==y for x,y in zip(l1.sts, l2.sts))

# get features
import math

MAX_DIMS = 16
MAX_BUFS = 9
def lin_to_feats(lin:Kernel, use_sts=True):
  assert lin.shape_len < MAX_DIMS, "too many dims"

  all_colors = ["blue", "cyan", "white", "green", "red", "magenta", "yellow"]
  lc = [all_colors.index(x) for x in lin.colors()]

  ret = []
  # before, some generic linearizer stuff
  ret.append(lin.upcasted)
  ret.append(lin.local_dims)

  # first, the full shape, including the colors
  for s,os,c in zip(lin.full_shape,lin.output_shape,lc):
    if isinstance(s, UOp):
      ret.append(False)
      ret += [0]*9
    else:
      ret.append(True)
      ret.append(math.log2(s))
      ret.append(min(33, s))
      ret.append(math.log2(os))
      ret.append(min(33, os))
      ret.append(s%2 == 0)
      ret.append(s%3 == 0)
      ret.append(s%4 == 0)
      ret.append(s%8 == 0)
      ret.append(s%16 == 0)
    cc = [0]*7
    cc[c] = 1
    ret += cc
  ret += [0] * (17*(MAX_DIMS-len(lin.full_shape)))
  ret = [float(x) for x in ret]

  if use_sts:
    my_sts = dedup([(x.shape == lin.full_shape, x.real_strides(), any(v.mask is not None for v in x.views), len(x.views)) for x in lin.sts])
    assert len(my_sts) < MAX_BUFS
    sts_len = 3 + 5*MAX_DIMS
    for s in my_sts:
      ret.append(s[0])  # reduce
      ret.append(s[2])  # has mask
      ret.append(s[3])  # len views
      for d in s[1]:
        ret.append(d is None)
        ret.append(d == 0)
        ret.append(d == 1)
        ret.append(min(33, d) if d is not None else -1)
        if d is not None and d >= 1: ret.append(math.log2(d))
        else: ret.append(-1)
      ret += [0] * (5*(MAX_DIMS - len(s[1])))
    ret += [0] * (sts_len*(MAX_BUFS - len(my_sts)))
    assert len(ret) == 1021, f"wrong len {len(ret)}"
  else:
    assert len(ret) == 274, f"wrong len {len(ret)}"
  return ret

from tinygrad.device import Device, Buffer
from tinygrad.engine.search import _ensure_buffer_alloc, _time_program
from tinygrad.helpers import to_function_name, CACHELEVEL, diskcache_get, diskcache_put

def time_linearizer(lin:Kernel, rawbufs:list[Buffer], allow_test_size=True, max_global_size=65536, cnt=3, disable_cache=False, clear_l2=False) -> float:  # noqa: E501
  key = {"ast": lin.ast.key, "opts": str(lin.applied_opts), "allow_test_size": allow_test_size,
         "max_global_size": max_global_size, "clear_l2": clear_l2, "device": lin.opts.device, "suffix": lin.opts.suffix}
  if not disable_cache and CACHELEVEL >= 2 and (val:=diskcache_get("time_linearizer", key)) is not None: return min(val)

  dev = Device[lin.opts.device]
  assert dev.compiler is not None

  rawbufs = _ensure_buffer_alloc(rawbufs)
  var_vals: dict[Variable, int] = {k:int(k.vmax+k.vmin)//2 for k in lin.ast.variables()}
  p = lin.to_program()
  tms = _time_program(p, dev.compiler.compile(p.src), var_vals, rawbufs,
                      max_global_size=max_global_size if allow_test_size else None, clear_l2=clear_l2, cnt=cnt, name=to_function_name(lin.name))

  if CACHELEVEL >= 2: diskcache_put("time_linearizer", key, tms)
  return min(tms)
