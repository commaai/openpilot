# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
from typing import Any, TYPE_CHECKING
import pickle, base64, itertools, time, sys, functools
from dataclasses import replace
from tinygrad.dtype import DType, dtypes, AddrSpace, truncate, storage_fmt_for_dtype, to_storage_scalar, from_storage_scalar
from tinygrad.helpers import all_same, getenv, flatten, Target, IMAGE, is_image_shape, cpu_profile
from tinygrad.device import Buffer, Compiled, Compiler, Allocator
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import exec_alu, python_alu, Ops, UOp, GroupOp, bitcast
from tinygrad.renderer import Renderer

def _load(m, i, dtype: DType):
  if i is None: return 0.0
  if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
  return from_storage_scalar(m[i], dtype)

def load(inp, j, dtype: DType):
  if len(inp) >= 3: return [_load(m, x+j if x is not None else None, dtype) if gate else default for (m,x),default,gate in zip(*inp[:3])]
  return [_load(m, x+j if x is not None else None, dtype) for m,x in inp[0]]

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
    self.uops: list[UOp] = pickle.loads(lib)
    self.uop_to_index: dict[UOp, int] = {u:i for i,u in enumerate(self.uops)}
    self.loop_ends: dict[UOp, int] = {u.src[1]:i for i, u in enumerate(self.uops) if u.op == Ops.END}
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    st = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    void_ops = {Ops.END, Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP, Ops.GROUP, Ops.STORE, Ops.LOOP}
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
      values: dict[UOp, Any] = {}
      pbufs: list[memoryview] = list(bufs)
      pvals: list[int] = list(vals)
      exec_masks = [[True] * warp_size]
      i = 0
      while i < len(self.uops):
        u = self.uops[i]
        src_values = [values[v] for v in u.src if v.op not in void_ops]
        src_dtypes = [v.dtype for v in u.src if v.op not in void_ops]
        if getenv("TRACE"): print(i, u.op, u.dtype, u.arg, src_values, src_dtypes)
        if u.op is Ops.END:
          if len(u.src) == 3:
            # conditional backedge on LOOP: jump back while the condition is true
            if values[u.src[2]][0]: i = self.uop_to_index[u.src[1]]
            else: i += 1
          else: i = self.uop_to_index[u.src[1]]
          continue
        if u.op is Ops.IF:
          exec_masks.append([x and y for x,y in zip(exec_masks[-1], src_values[0])])
          i += 1
          continue
        if u.op is Ops.ENDIF:
          exec_masks.pop()
          i += 1
          continue
        if u.op in (Ops.BARRIER, Ops.SINK, Ops.NOOP, Ops.GROUP, Ops.LOOP):
          # in the python emulator, the warp is always in sync
          i += 1
          continue
        assert u.dtype is not None, f"{u.op} is missing a dtype"
        if u.op is Ops.STORE:
          assert len(src_values) == 2, f"STORE must be lowered to 2 srcs, got {len(src_values)}"
          store_gate = exec_masks[-1]
          for j,val in enumerate(src_values[1] if u.max_numel() > 1 else [src_values[1]]):
            for (m,o),v,g in zip(src_values[0], val, store_gate):
              if g: _store(m, o+j, v, src_dtypes[1])
          i += 1
          continue
        if u.op is Ops.AFTER: values[u] = src_values[0]
        elif u.op is Ops.PARAM and u.addrspace is AddrSpace.ALU: values[u] = [pvals.pop(0)] * warp_size
        elif u.op in {Ops.PARAM, Ops.BUFFER}:
          storage_fmt = storage_fmt_for_dtype(u.dtype)
          if storage_fmt is None: raise RuntimeError(f"dtype={u.dtype} is not supported")
          if TYPE_CHECKING or sys.version_info < (3, 12): assert storage_fmt != "e"
          if u.addrspace == AddrSpace.REG:
            # REGs are per thread
            values[u] = [memoryview(bytearray(u.max_numel()*u.dtype.itemsize)).cast(storage_fmt) for _ in range(warp_size)]
          else:
            buf = memoryview(bytearray(u.max_numel()*u.dtype.itemsize)) if u.op is not Ops.PARAM else pbufs.pop(0)
            values[u] = [buf.cast(storage_fmt)] * warp_size
        elif u.op is Ops.SPECIAL:
          if u.arg[0] == 'g': values[u] = [idxs[2-int(u.arg[-1])]] * warp_size
          elif u.arg[0] == 'l': values[u] = [x[2-int(u.arg[-1])] for x in warp]
        elif u.op is Ops.CONST: values[u] = [u.arg] * warp_size
        elif u.op in {Ops.INDEX, Ops.SHRINK}:
          ret:list = []
          if u.src[0].addrspace == AddrSpace.ALU:
            ret = [src_values[0][i][t] for t,i in enumerate(src_values[1])]
          elif is_image_shape(u.src[0]._shape):
            for m,oy,ox in zip(*src_values):
              if ox < 0 or ox >= u.src[0]._shape[1] or oy < 0 or oy >= u.src[0]._shape[0]: ret.append((m, None))
              else: ret.append((m, ox*4 + oy*u.src[0]._shape[1]*4))
          else:
            for m,o in zip(src_values[0], src_values[1]): ret.append((m,o))
          values[u] = ret
        elif u.op is Ops.RANGE:
          if u not in values: values[u] = [0] * warp_size
          else:
            for j in range(len(values[u])):
              values[u][j] += 1
          if values[u][0] == src_values[0][0]:
            del values[u]
            i = self.loop_ends[u] + 1
            continue
        elif u.op is Ops.STACK: values[u] = src_values
        elif u.op is Ops.BITCAST: values[u] = [bitcast(x, src_dtypes[0], u.dtype) for x in src_values[0]]
        elif u.op is Ops.CAST:
          values[u] = [truncate.get(u.dtype, lambda dt: dt)(u.dtype.const(x)) for x in src_values[0]]
        elif u.op is Ops.LOAD:
          if (load_sz := u.max_numel()) > 1:
            # buf and gate are not vecs
            values[u] = [load([src_values[k] if k in [0,2] else src_values[k][j] \
                               for k in range(len(src_values))], j, u.dtype) for j in range(load_sz)]
          else:
            values[u] = load(src_values, 0, u.dtype)
        elif u.op is Ops.WMMA:
          first_src_dtype = u.src[0].dtype
          assert isinstance(first_src_dtype, DType) # mypy
          dims, dtype_in, device, threads = u.arg[0], first_src_dtype, u.arg[2], u.arg[3]
          wmma_helper = functools.partial(generic_wmma_helper, src_values, warp_size)
          # TODO: refactor these to a shared TensorCoreLayout
          if device == "METAL":
            # A (2 elements on 32 threads): row major
            def a_b_elem(x, i, j, goff): return x[(i%2)][goff+(i//2)%2+(j%4)*2+(i//4)*8+(j//4)*16]
            # (i, j), C, D (2 elements on 32 threads): row major same as A/B
            def c_map(lane, elem): return (elem + ((lane%2)*2) + ((lane//8)%2)*4, ((lane//2)%4) + (lane//16)*4)
            values[u] = wmma_helper(32, 8, 2, 2, 2, a_b_elem, a_b_elem, c_map)
          elif device == "AMD" and threads == 64:
            def a_elem(x, k, row, goff): return x[k%(dims[2]//4)][goff + (k//(dims[2]//4))*16 + row]
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)  # pylint: disable=arguments-out-of-order
            def c_map(lane, elem): return (lane%16, (lane//16)*4 + elem)
            values[u] = wmma_helper(64, dims[2], len(src_values[0]), len(src_values[1]), len(src_values[2]), a_elem, b_elem, c_map)
          elif device == "AMD" and len(src_values[0]) == 8: # RDNA4
            def a_elem(x, k, row, goff): return x[k - [0, 4, 4, 8][k//4]][goff + row + [0, 16, 0, 16][k//4]]
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)
            def c_map(lane, elem): return (lane%16, (lane//16)*8 + elem)
            values[u] = wmma_helper(32, 16, 8, 8, 8, a_elem, b_elem, c_map)
          elif device == "AMD":
            # A (16 elements on 32 threads): col major, lane 16-32 == lane 0-15
            def a_elem(x, k, row, goff):
              assert x[k][goff+row] == x[k][goff+row+16], "warp elements not duplicated properly across lanes"
              return x[k][goff+row]
            # B (16 elements on 32 threads): row major, lane 16-32 == lane 0-15
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)  # pylint: disable=arguments-out-of-order
            def c_map(lane, elem): return (lane%16, lane//16+elem*2) # (i, j), C, D (8 elements on 32 threads): row major
            values[u] = wmma_helper(32, 16, 16, 16, 8, a_elem, b_elem, c_map)
          elif device == "CUDA":
            # (col, row) given (lane, elem) for C & D (4 elements on 32 threads); shared by all tc shapes with M=16 N=8
            def c_map(lane, elem): return (elem%2 + (lane%4)*2, lane//4 + (elem//2)*8)

            if dims == (8,16,16):
              def a_elem(x, k, row, goff): return x[k%2 + (row//8)*2 + (k//8)*4][goff + (k//2)%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%2 + (k//8)*2][goff + (k//2)%4 + col*4]
              values[u] = wmma_helper(32, 16, 8, 4, 4, a_elem, b_elem, c_map)

            elif dims == (8,16,32):
              def a_elem(x, k, row, goff): return x[k%4 + (row//8)*4 + (k//16)*8][goff + (k//4)%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%4 + (k//16)*4][goff + (k//4)%4  + col*4]
              values[u] = wmma_helper(32, 32, 16, 8, 4, a_elem, b_elem, c_map)

            elif dims == (8,16,8) and dtype_in == dtypes.half:
              def a_elem(x, k, row, goff): return x[k%2 + (row//8)*2][goff + k//2 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%2][goff + k//2 + col*4]
              values[u] = wmma_helper(32, 8, 4, 2, 4, a_elem, b_elem, c_map)

            elif dims == (8,16,8) and dtype_in == dtypes.float:
              def a_elem(x, k, row, goff): return x[(k//4)*2 + row//8][goff + k%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k//4][goff + k%4 + col*4]
              values[u] = wmma_helper(32, 8, 4, 2, 4, a_elem, b_elem, c_map)

            else: raise NotImplementedError(f"unimplemented tensor core {u.arg}")
          else: raise NotImplementedError(f"unimplemented tensor core {u.arg}")
        elif u.op in GroupOp.ALU:
          assert all_same([len(x) for x in src_values]), f"{[len(x) for x in src_values]} doesn't match on {u.op}"
          assert all_same([u.dtype] + src_dtypes) or u.op in {*GroupOp.Comparison, Ops.WHERE}, f"dtype mismatch on {u.op}"
          values[u] = [exec_alu(u.op, u.dtype, p) for p in zip(*src_values)]
        assert u in values, u
        i += 1
    return time.perf_counter() - st

class PythonCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class PythonRenderer(Renderer):
  code_for_op = python_alu
  compiler = PythonCompiler()

  def __init__(self, target:Target):
    assert (emu:=getenv("EMULATE", "")) == "", ("EMULATE is deprecated, use DEV=PYTHON::" +
      {"AMD":"gfx1100", "AMD_RDNA4":"gfx1201", "AMD_MFMA":"gfx950", "CUDA":"sm_80", "CUDA_SM75":"sm_75", "CUDA_SM89":"sm_89"}.get(emu, emu))
    target = replace(target, renderer="PYTHON")
    if target.arch == "METAL": self.target, self.tensor_cores = replace(target, device="METAL"), tc.metal
    elif target.arch.startswith("gfx"):
      self.target = replace(target, device="AMD")
      self.tensor_cores = tc.get_amd(target.arch)
    elif target.arch.startswith("sm"):
      self.target = replace(target, device="CUDA")
      self.tensor_cores = tc.get_cuda(target.arch)
    elif IMAGE and not target.arch: self.target = replace(target, arch="IMAGE_PITCH_ALIGNMENT=1")
    else: self.target = target

  def render(self, uops:list[UOp]) -> str: return base64.b64encode(pickle.dumps(uops)).decode()

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d != dtypes.half or sys.version_info >= (3, 12)}

class PythonAllocator(Allocator['PythonDevice']):
  def _alloc(self, size, options): return memoryview(bytearray(size))
  def _as_buffer(self, src) -> memoryview: return src
  def _copyin(self, dest, src:memoryview):
    with cpu_profile("TINY -> PYTHON", f"{self.dev.device}:COPY"): dest[:] = src
  def _copyout(self, dest:memoryview, src):
    with cpu_profile("PYTHON -> TINY", f"{self.dev.device}:COPY"): dest[:] = src
  def map(self, buf:Buffer): return buf.as_memoryview(force_zero_copy=True)
  def _offset(self, buf:memoryview, size:int, offset:int): return buf[offset:offset+size]

class PythonDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, PythonAllocator(self), [PythonRenderer], PythonProgram)
