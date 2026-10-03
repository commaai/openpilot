# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
from typing import Any, TYPE_CHECKING
import pickle, base64, itertools, time, sys, ctypes
from dataclasses import replace
from tinygrad.dtype import bitcast, DType, dtypes, AddrSpace, truncate, storage_fmt_for_dtype, to_storage_scalar, from_storage_scalar
from tinygrad.helpers import all_same, getenv, Target, IMAGE, is_image_shape, to_mv, mv_address
from tinygrad.device import HostAllocator, Compiled, Compiler, Program, TinyELF
from tinygrad.renderer import tc
from tinygrad.uop.ops import exec_alu, python_alu, Ops, UOp, GroupOp
from tinygrad.renderer import Renderer

def _load(m, i, dtype: DType):
  if i is None: return 0.0
  if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
  if (w:=m.nbytes // len(m)) >= dtype.itemsize: return from_storage_scalar(m[i], dtype)
  return sum(m[i+k] << (8*w*k) for k in range(dtype.itemsize // w)) # a bitcast can read wider than the buffer, _store splits it the same way

def _step(m, dtype: DType): return max(1, dtype.itemsize // (m.nbytes // len(m))) # storage elements per lane

def load(inp, j, dtype: DType):
  if len(inp) >= 3: return [_load(m, x+j*_step(m, dtype) if x is not None else None, dtype) if gate else alt for (m,x),alt,gate in zip(*inp[:3])]
  return [_load(m, x+j*_step(m, dtype) if x is not None else None, dtype) for m,x in inp[0]]

def _store(m, i, v, dtype: DType):
  if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
  if (w:=m.nbytes // len(m)) >= dtype.itemsize: m[i] = to_storage_scalar(v, dtype)
  else:
    for k in range(dtype.itemsize // w): m[i+k] = (v >> 8*w*k) & ((1 << 8*w) - 1)

def wmma(tensor_cores:list[tc.TensorCore], arg, inp, warp_size:int):
  # cores sharing (dims, dtype_in, threads) share fragments, so the first match is the layout
  tcore = next(x for x in tensor_cores if (x.dims, x.dtype_in, x.threads) == arg[:3])
  frags = tcore.frag_coords()
  for cc,x,co in zip("ABC", inp, frags): assert len(x) == len(co[0]), f"{cc} must have {len(co[0])} elements per thread, it has {len(x)}"
  assert warp_size % tcore.threads == 0, f"must have multiples of {tcore.threads} warp threads"
  out = [x[:] for x in inp[2]]
  for goff in range(0, warp_size, tcore.threads):
    a, b = ({c: x[e][goff+lane] for lane,lc in enumerate(co) for e,c in enumerate(lc)} for co,x in zip(frags[:2], inp))
    for lane,lc in enumerate(frags[2]):
      for e,(m,n) in enumerate(lc): out[e][goff+lane] += sum(a[m,k]*b[k,n] for k in range(tcore.dims[2]))
  return out

class PythonProgram(Program['PythonDevice']):
  def __init__(self, dev:'PythonDevice', obj:TinyELF):
    self.uops: list[UOp] = pickle.loads(obj.lib)
    self.tensor_cores = PythonRenderer(obj.target).tensor_cores
    self.uop_to_index: dict[UOp, int] = {u:i for i,u in enumerate(self.uops)}
    self.loop_ends: dict[UOp, int] = {u.src[1]:i for i, u in enumerate(self.uops) if u.op == Ops.END}
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    st = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
      values: dict[UOp, Any] = {}
      pbufs: list[int] = list(bufs)
      pvals: list[int] = list(vals)
      exec_masks = [[True] * warp_size]
      i = 0
      while i < len(self.uops):
        u = self.uops[i]
        src_values = [values[v] for v in u.src if v.dtype is not dtypes.void]
        src_dtypes = [v.dtype for v in u.src if v.dtype is not dtypes.void]
        if getenv("TRACE"): print(i, u.op, u.dtype, u.arg, src_values, src_dtypes)
        if u.op is Ops.END:
          if len(u.src) == 3:
            # conditional backedge on a loop: jump back while the condition is true
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
        if u.op in (Ops.BARRIER, Ops.SINK, Ops.NOOP, Ops.GROUP, Ops.CUSTOM_FUNCTION) or (u.op is Ops.RANGE and u.dtype == dtypes.void):
          # in the python emulator, the warp is always in sync
          i += 1
          continue
        if u.op is Ops.STORE:
          assert len(src_values) == 2, f"STORE must be lowered to 2 srcs, got {len(src_values)}"
          store_gate = exec_masks[-1]
          for j,val in enumerate(src_values[1] if u.max_numel() > 1 else [src_values[1]]):
            for (m,o),v,g in zip(src_values[0], val, store_gate):
              if g: _store(m, o+j*_step(m, src_dtypes[1]), v, src_dtypes[1])
          i += 1
          continue
        if u.op is Ops.AFTER or (u.op is Ops.BITCAST and u.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL)): values[u] = src_values[0]
        elif u.op is Ops.PARAM and u.addrspace is AddrSpace.ALU: values[u] = [pvals.pop(0)] * warp_size
        elif u.op in {Ops.PARAM, Ops.BUFFER}:
          storage_fmt = storage_fmt_for_dtype(u.dtype)
          if storage_fmt is None: raise RuntimeError(f"dtype={u.dtype} is not supported")
          if TYPE_CHECKING or sys.version_info < (3, 12): assert storage_fmt != "e"
          if u.addrspace == AddrSpace.REG:
            # REGs are per thread
            values[u] = [memoryview(bytearray(u.max_numel()*u.dtype.itemsize)).cast(storage_fmt) for _ in range(warp_size)]
          else:
            size = u.max_numel() * u.dtype.itemsize
            buf = memoryview(bytearray(size)) if u.op is not Ops.PARAM else to_mv(pbufs.pop(0), size)
            values[u] = [buf.cast(storage_fmt)] * warp_size
        elif u.op is Ops.SPECIAL:
          if u.arg[0] == 'g': values[u] = [idxs[2-int(u.arg[-1])]] * warp_size
          elif u.arg[0] == 'l': values[u] = [x[2-int(u.arg[-1])] for x in warp]
        elif u.op is Ops.CONST: values[u] = [u.val] * warp_size
        elif u.op in {Ops.INDEX, Ops.SHRINK}:
          ret:list = []
          if u.src[0].addrspace == AddrSpace.ALU:
            ret = [src_values[0][i][t] for t,i in enumerate(src_values[1])]
          elif is_image_shape(u.src[0]._shape):
            for m,oy,ox in zip(*src_values):
              if ox < 0 or ox >= u.src[0]._shape[1] or oy < 0 or oy >= u.src[0]._shape[0]: ret.append((m, None))
              else: ret.append((m, ox*4 + oy*u.src[0]._shape[1]*4))
          else:
            scale = u.src[0].dtype.itemsize // u.src[0].src[0].dtype.itemsize if u.src[0].op is Ops.BITCAST else 1
            for m,o in zip(src_values[0], src_values[1]): ret.append((m[0], m[1]+o*scale) if isinstance(m, tuple) else (m, o*scale))
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
        elif u.op is Ops.CALL:
          restype = None if u.dtype is dtypes.void else getattr(ctypes, f"c_{'u' if u.dtype in dtypes.uints else ''}int{u.dtype.bitsize}")
          cfunc = ctypes.CFUNCTYPE(restype, *[ctypes.c_uint64] * len(src_values))
          values[u] = []
          for fptr,args,gate in zip(values[u.src[0].src[0]], zip(*src_values), exec_masks[-1]):
            call_args = [(mv_address(x[0]) + x[1]*dt.itemsize) if isinstance(x, tuple) else x for x,dt in zip(args, src_dtypes)]
            values[u].append(cfunc(fptr)(*call_args) if gate else None)
        elif u.op is Ops.WMMA: values[u] = wmma(self.tensor_cores, u.arg, src_values, warp_size)
        elif u.op in GroupOp.ALU:
          assert all_same([len(x) for x in src_values]), f"{[len(x) for x in src_values]} doesn't match on {u.op}"
          assert all_same([u.dtype] + src_dtypes) or u.op in {*GroupOp.Comparison, Ops.WHERE, Ops.SHL, Ops.SHR}, f"dtype mismatch on {u.op}"
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

class PythonDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, HostAllocator(self), [PythonRenderer], PythonProgram)
