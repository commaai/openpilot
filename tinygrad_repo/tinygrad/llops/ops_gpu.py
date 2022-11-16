from __future__ import annotations
import os, functools
import numpy as np
import pyopencl as cl  # type: ignore
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Union, Set
from tinygrad.helpers import prod, ConvArgs
from tinygrad.ops import DEBUG, UnaryOps, BinaryOps, ReduceOps, MovementOps
from tinygrad.shapetracker import ShapeTracker

CLCACHE = int(os.getenv("CLCACHE", "1"))
class CLBuffer:
  def __init__(self, size):
    if len(CL.BUFFER_CACHE[size]) > 0:
      self.cl = CL.BUFFER_CACHE[size].pop()
    else:
      # TODO: on GPU OOM, clear the cache
      self.cl = cl.Buffer(CL().cl_ctx, cl.mem_flags.READ_WRITE, size)
      CL.mem_used += self.cl.size

  def __del__(self):
    if CLCACHE:
      CL.BUFFER_CACHE[self.cl.size].append(self.cl)
    else:
      CL.mem_used -= self.cl.size

class CL:
  CACHE, kernel_count, mem_used, time_sum, ops_sum = None, -1, 0, 0.0, 0.0
  BUFFER_CACHE : Dict[int, List[cl.Buffer]] = defaultdict(list)
  cl_ctx : Optional[cl.Context] = None
  cl_queue : Optional[cl.CommandQueue] = None
  def __init__(self):
    if CL.cl_queue is not None:  # already initted
      return
    devices = sum([x.get_devices(device_type=cl.device_type.GPU) for x in cl.get_platforms()], [])
    if len(devices) == 0:  # settle for CPU
      devices = sum([x.get_devices(device_type=cl.device_type.CPU) for x in cl.get_platforms()], [])
    CL.cl_ctx = cl.Context(devices=[devices[int(os.getenv("CL_DEVICE", "0"))]])
    if len(devices) > 1 or DEBUG >= 1:
      print(f"using {CL.cl_ctx.devices}")
    CL.cl_queue = cl.CommandQueue(self.cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)  # this is an in-order command queue

  @staticmethod
  def enqueue_copy(a, b, is_blocking=False):
    if CL.CACHE is not None:
      assert False, "can't copy while caching"
    if DEBUG >= 1:
      print(f"**CL**        copy in {b.shape}" if isinstance(b, np.ndarray) else f"**CL**        copy OUT {a.shape}")
    cl.enqueue_copy(CL().cl_queue, a, b, is_blocking=is_blocking)

@functools.lru_cache(maxsize=None)
class CLProgram:
  kernel_cnt = 0
  def __init__(self, name:str, prg:str, options:Tuple[str, ...]=tuple(), argdtypes=None, rename=True, binary=False):
    self.name, self.prg, self.options, self.argdtypes = f"{name}_{CLProgram.kernel_cnt}" if rename else name, prg.replace(f"{name}(", f"{name}_{CLProgram.kernel_cnt}(") if rename else prg, options, argdtypes
    self.clprogram = cl.Program(CL().cl_ctx, CL().cl_ctx.devices, [self.prg]) if binary else cl.Program(CL().cl_ctx, self.prg)  # type: ignore
    self.clprg = self.clprogram.build(options=list(self.options)).__getattr__(self.name)
    if self.argdtypes is not None:
      self.clprg.set_scalar_arg_dtypes(self.argdtypes)
    CLProgram.kernel_cnt += 1
  def __call__(self, *args, op_estimate=0):
    CL.kernel_count += 1
    if CL.CACHE is not None:
      CL.CACHE.append((self, args))
    else:
      e = self.clprg(CL().cl_queue, *args)
    if DEBUG >= 4:
      print(self.prg)
    if DEBUG >= 2:
      CL.cl_queue.finish()
    if DEBUG >= 1:
      CL.time_sum += 0 if DEBUG <= 1 or CL.CACHE is not None else (e.profile.end - e.profile.start)
      CL.ops_sum += op_estimate
      print(f"**CL** {CL.kernel_count:6d} {self.name:20s} args {len(args[2:]):5d}  kernels {str(args[0]):18s} {str(args[1]):12s} OPs {op_estimate/1e6:7.1f}M/{CL.ops_sum/1e9:7.2f}G  mem {CL.mem_used/1e9:5.2f} GB " +
            ("" if DEBUG <= 1 or CL.CACHE is not None else f"tm {(e.profile.end - e.profile.start)/1e3:9.2f}us/{CL.time_sum/1e6:9.2f}ms ({op_estimate/(e.profile.end - e.profile.start):8.2f} GFLOPS)"))

# **** end CL wrappers ****

class GPUBuffer:
  code_for_op = {
    UnaryOps.NOOP: "(A)", UnaryOps.NEG: "(-(A))", UnaryOps.RELU: "max(A, (float)0.)",
    UnaryOps.EXP: "exp(A)", UnaryOps.LOG: "log(A)", UnaryOps.SIGN: "sign(A)", UnaryOps.RECIPROCAL: "((float)1.0/A)",
    BinaryOps.ADD: "(A+B)", BinaryOps.SUB: "(A-B)", BinaryOps.MUL: "(A*B)",
    BinaryOps.DIV: "(A/B)", BinaryOps.POW: "pow(A,B)", BinaryOps.CMPEQ: "(A==B)",
    ReduceOps.SUM: "(acc + A)", ReduceOps.MAX: "max(A, acc)"
  }
  start_for_op = {ReduceOps.SUM: "0.0", ReduceOps.MAX: "-INFINITY"}

  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf:Optional[GPUBuffer]=None, backing:Optional[np.ndarray]=None):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape
    self._buf : Optional[CLBuffer] = hostbuf._buf if hostbuf is not None else None
    self._base_shape : Tuple[int, ...] = hostbuf._base_shape if hostbuf is not None else self.shape
    self._backing : Optional[np.ndarray] = hostbuf._backing if hostbuf is not None else backing
    # early copy in for large buffers
    if self._backing is not None and self._backing.shape != (1,):
      self.cl
  
  @property
  def cl(self):
    if self._buf is None:
      self._buf = CLBuffer(4*prod(self._base_shape))
    if self._backing is not None:
      CL.enqueue_copy(self._buf.cl, self._backing, is_blocking=False)
      self._backing = None
    return self._buf.cl

  def __repr__(self): return f"<GPUBuffer with shape {self.shape!r}>"

  @staticmethod
  def fromCPU(x): return GPUBuffer(x.shape, backing=x.view(np.ndarray).astype(np.float32).ravel())

  def toCPU(self):
    data = np.empty(self.shape, dtype=np.float32)
    CL.enqueue_copy(data, self.contiguous_op().cl, is_blocking=True)
    return data

  def contiguous_view_constant_fold(x, name:str, reduce:Optional[int]=None) -> Tuple[str, Optional[str], str]:
    idx_getter = f"int valid = 1; {'long' if prod(x.shape) >= 2**31 else 'int'} idx = gid; {'idx *= '+str(reduce)+'; idx += subidx;' if reduce is not None else ''} {x.st.expr().replace('//', '/')};"
    constant = x._backing[0] if x._base_shape == (1,) and x._backing is not None else None
    args = (["__global const float *x"] if constant is None else []) + ["int gid"] + (["int subidx"] if reduce is not None else []) 
    return f"inline float get_{name}({','.join(args)}) {{ {idx_getter} return valid ? {constant if constant is not None else 'x[idx]'} : 0.0;}}", \
      f"__global const float *{name}_g" if constant is None else None, \
      f"get_{name}({name+'_g, ' if constant is None else ''}gid{', subidx' if reduce is not None else ''});"

  def unary_op(x, op:UnaryOps): return type(x)(x.shape)._processing_op([("A", x)], GPUBuffer.code_for_op[op])
  def binary_op(x, op:BinaryOps, y:GPUBuffer): return type(x)(x.shape)._processing_op([("A", x), ("B", y)], GPUBuffer.code_for_op[op])
  def contiguous_op(x): return x if x.st.contiguous else x.unary_op(UnaryOps.NOOP)
  def movement_op(x, op:MovementOps, arg) -> GPUBuffer: return type(x)(ShapeTracker(x.st).movement_op(op, arg), x)
  def reduce_op(x, op:ReduceOps, new_shape:Tuple[int, ...]): return type(x)(new_shape)._processing_op([("A", x)], code="acc", earlycode=GPUBuffer.code_for_op[op], earlybufs=set("A"), op=op)

  def _processing_op(ret, bufs: List[Tuple[str, GPUBuffer]]=[], code:str="acc", C:Optional[ConvArgs]=None, op=ReduceOps.SUM, reduce_shape=None, earlybufs:Set[str]=set(), earlycode:str="acc") -> GPUBuffer:
    assert C is None

    # get the input/output shape and the reduce amount
    reduce_shape = (bufs[0][1].shape, ret.shape) if reduce_shape is None else reduce_shape
    red = prod([s for s,n in zip(*reduce_shape) if n == 1])
    assert red < 2**31, f"reduce must be under 2**31, {red} isn't"

    # if it's a partial reduce, assert last non reduced axis is before the first reduced axis
    if red > 1 and prod(ret.shape) != 1:
      assert max([i for i,(s,n) in enumerate(zip(*reduce_shape)) if s == n and n != 1]) < min([i for i,(s,n) in enumerate(zip(*reduce_shape)) if s != 1 and n == 1])

    kernel_name = "reduce" if red > 1 else "elementwise"
    early_views = {name:buf.contiguous_view_constant_fold(name, red) for name, buf in bufs if name in earlybufs}
    late_views = {name:buf.contiguous_view_constant_fold(name) for name, buf in bufs if name not in earlybufs}
    views = {**early_views, **late_views}

    buf_types : List[str] = [views[name][1] for name, _ in bufs if views[name][1] is not None]  # type: ignore
    buf_cl = [buf.cl if 'image2d_t' not in views[name][1] else buf.image for name, buf in bufs if views[name][1] is not None]  # type: ignore

    # use local memory if it's a multistage reduce
    inter_red = 256 if (prod(ret.shape) < 8192 and red >= 256) else 1
    if inter_red > 1:
      buf_cl.append(cl.LocalMemory(inter_red*4))

    reduce_loop = f"int mid = get_global_id(1); for (int subidx = {red//inter_red + 1} * mid; subidx < min({red}, {red//inter_red + 1} * (mid+1)); subidx++)" if inter_red > 1 else f"for (int subidx = 0; subidx < {red}; subidx++)"
    conv_prg = CLProgram(kernel_name, f"""{chr(10).join([x[0] for x in views.values()])}
    __kernel void {kernel_name}({','.join(["__global float* restrict output"] + buf_types + (["__local float *temp"] if inter_red > 1 else []))}) {{
      const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
      float acc = {GPUBuffer.start_for_op[op]};
      int gid = get_global_id(0);
      {reduce_loop} {{
{chr(10).join([f'        float {name} = ' + early_views[name][2] for name in early_views])}
        acc = {earlycode};
      }}"""+(f"""
      temp[mid] = acc; barrier(CLK_LOCAL_MEM_FENCE);
      if (mid == 0) {{ acc = {GPUBuffer.start_for_op[op]};
        for (int rdx = 0; rdx < {inter_red}; rdx++) {{
          acc = {GPUBuffer.code_for_op[op].replace('A', 'temp[rdx]')};
        }}""" if inter_red != 1 else "{")+f"""
{chr(10).join([f'        float {name} = ' + late_views[name][2] for name in late_views])}
        output[gid] = {code};
      }}
    }}""")

    conv_prg([prod(ret.shape), inter_red, 1], [1, inter_red, 1] if inter_red > 1 else None, ret.cl, *buf_cl, op_estimate=prod(reduce_shape[0])*len(earlybufs) + prod(reduce_shape[1])*len(bufs))
    return ret
