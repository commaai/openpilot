# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
from typing import Any, TYPE_CHECKING
import pickle, base64, itertools, time, struct, sys, functools
from tinygrad.dtype import DType, dtypes, ImageDType, PtrDType, truncate, float_to_fp16, float_to_bf16, float_to_fp8, fp8_to_float
from tinygrad.helpers import all_same, getenv, flatten, get_single_element, EMULATE
from tinygrad.device import Compiled, Compiler, Allocator, CompilerSet, CompilerPair
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import exec_alu, python_alu, Ops, UOp, GroupOp
from tinygrad.renderer import Renderer

def storage_fmt_for_dtype(dtype: DType): return 'H' if dtype == dtypes.bfloat16 else 'B' if dtype in dtypes.fp8s else dtype.fmt

def to_storage_scalar(x, dtype: DType):
  if dtype == dtypes.half: return float_to_fp16(x)
  if dtype == dtypes.bfloat16: return (struct.unpack('I', struct.pack('f', float_to_bf16(x)))[0] >> 16) & 0xFFFF
  if dtype in dtypes.fp8s: return float_to_fp8(float(x), dtype)
  return x

def from_storage_scalar(x, dtype: DType):
  if dtype == dtypes.bfloat16: return struct.unpack('f', struct.pack('I', (x & 0xFFFF) << 16))[0]
  if dtype in dtypes.fp8s: return fp8_to_float(int(x), dtype)
  return x

def _load(m, i, dtype: DType):
  if i is None: return 0.0
  if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
  return from_storage_scalar(m[i], dtype)

def load(inp, j, dtype: DType):
  if len(inp) == 2: return [_load(m, x+j if x is not None else None, dtype) if gate else default for (m,x,gate),default in zip(*inp)]
  return [_load(m, x+j if x is not None else None, dtype) for m,x,_ in inp[0]]

def _store(m, i, v, dtype: DType):
  if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
  m[i] = to_storage_scalar(v, dtype)

# here are the models for the WMMA instruction on the different hardware
def generic_wmma_helper(inp, warp_size, WARP_THREADS, K, NUM_A, NUM_B, NUM_C, a_elem, b_elem, c_map):
  for cc, tinp, num in zip(("A", "B", "C"), inp, (NUM_A, NUM_B, NUM_C)):
    assert len(tinp) == num, f"{cc} must have {num} elements per thread, it has {len(tinp)}"
    assert len(flatten(tinp)) == num * warp_size, f"WMMA must have {num * warp_size} total elements for {cc} in WMMA"
  assert warp_size > 0 and warp_size % WARP_THREADS == 0, f"must have multiples of {WARP_THREADS} warp threads"
  out = [inp[2][elem_idx][:] for elem_idx in range(NUM_C)]
  for goff in range(0, warp_size, WARP_THREADS):
    for lane_id in range(WARP_THREADS):
      for elem_idx in range(NUM_C): # calculate new muls and add to acc
        (c_i, c_j) = c_map(lane_id, elem_idx)
        out[elem_idx][goff+lane_id] += sum(a_elem(inp[0], _k, c_j, goff) * b_elem(inp[1], c_i, _k, goff) for _k in range(K))
  return out

class PythonProgram:
  def __init__(self, name:str, lib:bytes, **kwargs):
    self.uops: list[tuple[Ops, DType, list[int], Any]] = pickle.loads(lib)
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    st = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    void_ops = {Ops.END, Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP, Ops.GROUP, Ops.STORE}
    loop_ends: dict[int, int] = {srcs[1]:i for i, (uop, _, srcs, _) in enumerate(self.uops) if uop == Ops.END}
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
      values: dict[int, Any] = {}
      pbufs: list[memoryview] = list(bufs)
      pvals: list[int] = list(vals)
      i = 0
      while i < len(self.uops):
        uop, dtype, srcs, arg = self.uops[i]
        src_values = [values[v] for v in srcs if self.uops[v][0] not in void_ops]
        src_dtypes = [self.uops[v][1] for v in srcs if self.uops[v][0] not in void_ops]
        if getenv("TRACE"): print(i, uop, dtype, arg, src_values, src_dtypes)
        if uop is Ops.END:
          i = srcs[1]
          continue
        if uop in (Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP, Ops.GROUP):
          # in the python emulator, the warp is always in sync
          i += 1
          continue
        assert dtype is not None, f"{uop} is missing a dtype"
        if uop is Ops.STORE:
          for j,val in enumerate(src_values[1] if src_dtypes[1].count > 1 else [src_values[1]]):
            for (m,o,g),v in zip(src_values[0], val):
              if g: _store(m, o+j, v, src_dtypes[1].scalar())
          i += 1
          continue
        if uop is Ops.AFTER: values[i] = src_values[0]
        elif uop in {Ops.PARAM, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}:
          assert isinstance(dtype, PtrDType), dtype
          storage_fmt = storage_fmt_for_dtype(dtype.base.scalar())
          if storage_fmt is None: raise RuntimeError(f"{dtype=} is not supported")
          if TYPE_CHECKING or sys.version_info < (3, 12): assert storage_fmt != "e"
          if uop is Ops.DEFINE_REG:
            # REGs are per thread
            values[i] = [memoryview(bytearray(dtype.size*dtype.itemsize)).cast(storage_fmt) for _ in range(warp_size)]
          else:
            buf = memoryview(bytearray(dtype.size*dtype.itemsize)) if uop is not Ops.PARAM else pbufs.pop(0)
            values[i] = [buf.cast(storage_fmt)] * warp_size
        elif uop is Ops.DEFINE_VAR:
          values[i] = [pvals.pop(0)] * warp_size
        elif uop is Ops.SPECIAL:
          if arg[0] == 'g': values[i] = [idxs[2-int(arg[-1])]] * warp_size
          elif arg[0] == 'l': values[i] = [x[2-int(arg[-1])] for x in warp]
        elif uop is Ops.CONST: values[i] = [arg] * warp_size
        elif uop is Ops.INDEX:
          ret:list = []
          if isinstance(src_dtypes[0], ImageDType):
            for m,ox,oy in zip(src_values[0], src_values[1][0], src_values[1][1]):
              if ox < 0 or ox >= src_dtypes[0].shape[1] or oy < 0 or oy >= src_dtypes[0].shape[0]: ret.append((m, None))
              else: ret.append((m, ox*4 + oy*src_dtypes[0].shape[1]*4))
          else:
            for m,o in zip(src_values[0], src_values[1]): ret.append((m,o))
          values[i] = [(m,o,g) for (m,o),g in zip(ret, src_values[2] if len(src_values) == 3 else [True]*len(ret))] # set the gate last
        elif uop is Ops.CAST and isinstance(dtype, PtrDType):
          values[i] = src_values[0]
        elif uop is Ops.RANGE:
          if i not in values: values[i] = [0] * warp_size
          else:
            for j in range(len(values[i])):
              values[i][j] += 1
          if values[i][0] == src_values[0][0]:
            del values[i]
            i = loop_ends[i] + 1
            continue
        elif uop is Ops.VECTORIZE: values[i] = src_values
        elif uop is Ops.BITCAST:
          packed = struct.pack(str(warp_size) + storage_fmt_for_dtype(src_dtypes[0].scalar()),
                               *[to_storage_scalar(x, src_dtypes[0].scalar()) for x in src_values[0]])
          values[i] = list(struct.unpack(str(warp_size) +  storage_fmt_for_dtype(dtype.scalar()), packed))
          values[i] = [from_storage_scalar(x, dtype.scalar()) for x in values[i]]
        elif uop is Ops.CAST:
          values[i] = [truncate.get(dtype, lambda dt: dt)(dtypes.as_const(x, dtype)) for x in src_values[0]]
        elif uop is Ops.LOAD:
          if dtype.count > 1:
            values[i] = [load([src_values[i][j] if i != 0 and src_dtypes[i].count > 1 else src_values[i] \
                               for i in range(len(src_values))], j, dtype.scalar()) for j in range(dtype.count)]
          else:
            values[i] = load(src_values, 0, dtype)
        elif uop is Ops.GEP: values[i] = src_values[0][get_single_element(arg)]
        elif uop is Ops.WMMA:
          first_src_dtype = self.uops[srcs[0]][1]
          assert isinstance(first_src_dtype, DType) # mypy
          dims, dtype_in, device, threads = arg[1], first_src_dtype.scalar(), arg[4], arg[5]
          wmma_helper = functools.partial(generic_wmma_helper, src_values, warp_size)
          # TODO: refactor these to a shared TensorCoreLayout
          if device == "METAL":
            # A (2 elements on 32 threads): row major
            def a_b_elem(x, i, j, goff): return x[(i%2)][goff+(i//2)%2+(j%4)*2+(i//4)*8+(j//4)*16]
            # (i, j), C, D (2 elements on 32 threads): row major same as A/B
            def c_map(lane, elem): return (elem + ((lane%2)*2) + ((lane//8)%2)*4, ((lane//2)%4) + (lane//16)*4)
            values[i] = wmma_helper(32, 8, 2, 2, 2, a_b_elem, a_b_elem, c_map)
          elif device == "AMD" and threads == 64:
            def a_elem(x, k, row, goff): return x[k%(dims[2]//4)][goff + (k//(dims[2]//4))*16 + row]
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)  # pylint: disable=arguments-out-of-order
            def c_map(lane, elem): return (lane%16, (lane//16)*4 + elem)
            values[i] = wmma_helper(64, dims[2], len(src_values[0]), len(src_values[1]), len(src_values[2]), a_elem, b_elem, c_map)
          elif device == "AMD" and len(src_values[0]) == 8: # RDNA4
            def a_elem(x, k, row, goff): return x[k - [0, 4, 4, 8][k//4]][goff + row + [0, 16, 0, 16][k//4]]
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)
            def c_map(lane, elem): return (lane%16, (lane//16)*8 + elem)
            values[i] = wmma_helper(32, 16, 8, 8, 8, a_elem, b_elem, c_map)
          elif device == "AMD":
            # A (16 elements on 32 threads): col major, lane 16-32 == lane 0-15
            def a_elem(x, k, row, goff):
              assert x[k][goff+row] == x[k][goff+row+16], "warp elements not duplicated properly across lanes"
              return x[k][goff+row]
            # B (16 elements on 32 threads): row major, lane 16-32 == lane 0-15
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)  # pylint: disable=arguments-out-of-order
            def c_map(lane, elem): return (lane%16, lane//16+elem*2) # (i, j), C, D (8 elements on 32 threads): row major
            values[i] = wmma_helper(32, 16, 16, 16, 8, a_elem, b_elem, c_map)
          elif device == "CUDA":
            # (col, row) given (lane, elem) for C & D (4 elements on 32 threads); shared by all tc shapes with M=16 N=8
            def c_map(lane, elem): return (elem%2 + (lane%4)*2, lane//4 + (elem//2)*8)

            if dims == (8,16,16):
              def a_elem(x, k, row, goff): return x[k%2 + (row//8)*2 + (k//8)*4][goff + (k//2)%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%2 + (k//8)*2][goff + (k//2)%4 + col*4]
              values[i] = wmma_helper(32, 16, 8, 4, 4, a_elem, b_elem, c_map)

            elif dims == (8,16,32):
              def a_elem(x, k, row, goff): return x[k%4 + (row//8)*4 + (k//16)*8][goff + (k//4)%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%4 + (k//16)*4][goff + (k//4)%4  + col*4]
              values[i] = wmma_helper(32, 32, 16, 8, 4, a_elem, b_elem, c_map)

            elif dims == (8,16,8) and dtype_in == dtypes.half:
              def a_elem(x, k, row, goff): return x[k%2 + (row//8)*2][goff + k//2 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%2][goff + k//2 + col*4]
              values[i] = wmma_helper(32, 8, 4, 2, 4, a_elem, b_elem, c_map)

            elif dims == (8,16,8) and dtype_in == dtypes.float:
              def a_elem(x, k, row, goff): return x[(k//4)*2 + row//8][goff + k%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k//4][goff + k%4 + col*4]
              values[i] = wmma_helper(32, 8, 4, 2, 4, a_elem, b_elem, c_map)

            else: raise NotImplementedError(f"unimplemented tensor core {arg}")
          elif device == "INTEL":
            # A (16 elements on 8 threads)
            def a_elem(x, k, row, goff): return x[k%2+row*2][goff+k//2]
            # B (16 elements on 8 threads)
            def b_elem(x, col, k, goff): return x[k][goff+col]
            # C, D (8 elements on 8 threads)
            def c_map(lane, elem): return (lane, elem)
            values[i] = wmma_helper(8, 16, 16, 16, 8, a_elem, b_elem, c_map)
          elif device == "CPU":
            def elem(x, col, row, _): return x[col+row][0] # k is always 0
            def c_map(lane, elem): return (elem%16, elem//16)
            values[i] = wmma_helper(1, 1, 16, 16, 256, elem, elem, c_map)
          else: raise NotImplementedError(f"unimplemented tensor core {arg}")
        elif uop in GroupOp.ALU:
          assert all_same([len(x) for x in src_values]), f"{[len(x) for x in src_values]} doesn't match on {uop}"
          assert all_same([dtype] + src_dtypes) or uop in {*GroupOp.Comparison, Ops.WHERE}, f"dtype mismatch on {uop}"
          values[i] = [exec_alu(uop, dtype, p) for p in zip(*src_values)]
        assert i in values, (uop, dtype, srcs, arg)
        i += 1
    return time.perf_counter() - st

class PythonRenderer(Renderer):
  device = "PYTHON"
  code_for_op = python_alu
  def __init__(self):
    match EMULATE.value:
      case "METAL": self.device, self.tensor_cores = "METAL", tc.metal
      case "AMD": self.device, self.tensor_cores = "AMD", tc.amd_rdna3
      case "AMD_MFMA": self.device, self.tensor_cores = "AMD", tc.amd_cdna4
      case "AMD_RDNA4": self.device, self.tensor_cores = "AMD", tc.amd_rdna4
      case "CUDA": self.device, self.tensor_cores = "CUDA", tc.cuda_sm80
      case "CUDA_SM75": self.device, self.tensor_cores = "CUDA", tc.cuda_sm75
      case "CUDA_SM89": self.device, self.tensor_cores = "CUDA", tc.cuda_sm89
      case "INTEL": self.device, self.suffix, self.tensor_cores = "INTEL", "INTEL", tc.intel
      case "AMX": self.device, self.tensor_cores = "CPU", tc.amx
      case "": pass
      case _: raise RuntimeError(f"can't EMULATE device: {EMULATE.value}")

  def render(self, uops:list[UOp]) -> str:
    # the value of SPECIAL comes from local/global_size, not form its source
    lops = [(u.op, u.dtype, [uops.index(v) for v in u.src if u.op is not Ops.SPECIAL], u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(lops)).decode()

class PythonCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class PythonAllocator(Allocator['PythonDevice']):
  def _alloc(self, size, options): return memoryview(bytearray(size))
  def _copyin(self, dest, src:memoryview): dest[:] = src
  def _copyout(self, dest:memoryview, src): dest[:] = src

class PythonDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, PythonAllocator(self), CompilerSet([CompilerPair(PythonRenderer, PythonCompiler)]), PythonProgram)
