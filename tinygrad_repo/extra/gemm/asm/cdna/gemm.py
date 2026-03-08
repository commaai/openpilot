import atexit, functools
from tinygrad.runtime.support.compiler_amd import HIPCompiler
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.renderer import Estimates
from tinygrad.helpers import getenv, all_same, dedup
from extra.gemm.asm.cdna.asm import build_kernel, GEMM_ARGS

# ** CDNA4 assembly gemm

WORKGROUP_SIZE = 256

def custom_asm_gemm(C:UOp, A:UOp, B:UOp, dname:str, arch:str, wg:int) -> UOp:
  batch, M, K = A.shape
  K2, N = B.shape[(1 if B.ndim == 3 else 0):]
  assert K == K2
  lidx = UOp.special(WORKGROUP_SIZE, "lidx0")
  gidx = UOp.special(wg, "gidx0")
  k = build_kernel(batch, M, N, K, A.dtype.base)
  sink = UOp.sink(C.base, A.base, B.base, lidx, gidx,
                  arg=KernelInfo(name=k.name, estimates=Estimates(ops=2*batch*M*N*K, mem=(batch*M*K + K*N + batch*M*N)*2)))
  binary = HIPCompiler(arch).compile(k.to_asm())
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=k.to_text()), UOp(Ops.BINARY, arg=binary)))

counters = {"used":0, "todos":[]}
def todo(msg:str) -> bool: counters["todos"].append(msg); return False
atexit.register(lambda: print(f'asm_gemm: {counters["used"]} used, {len(counters["todos"])} not used'))

def can_use_asm_gemm(a:Tensor, b:Tensor) -> bool:
  if a.dtype != b.dtype: return todo(f"dtypes must match {a.dtype} != {b.dtype}")
  if a.dtype not in {dtypes.bfloat16, dtypes.float16}: return todo(f"only bfloat16/float16, got {a.dtype}")
  # only sharding on the batch is tested, others might work too
  if isinstance(a.device, tuple) and not (a.ndim == 3 and a.uop.axis == 0 and b.uop.axis is None):
    return todo(f"sharding mismatch a.ndim={a.ndim} a.uop.axis={a.uop.axis} b.uop.axis={b.uop.axis}")
  batch, M, K = (1, *a.shape) if a.ndim == 2 else a.shape
  N = b.shape[1]
  if isinstance(a.device, tuple): batch //= len(a.device)
  if batch not in {1, 2}: return todo(f"GEMM batch size {batch}")
  if (key:=(M, N, K)) not in GEMM_ARGS: return todo(f"GEMM shape not supported {key}")
  return True

# ** UOp gemm to test Tensor.custom_kernel multi and backward correctness on non cdna4
# note: this can be removed after we have GEMM on mixins

def custom_uop_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  M, K = A.shape[0]*A.shape[1], A.shape[2]
  K2, N = B.shape[(1 if B.ndim == 3 else 0):]
  assert K == K2
  m = UOp.range(M, 1, AxisType.LOOP)
  n = UOp.range(N, 2, AxisType.LOOP)
  k = UOp.range(K, 0, AxisType.REDUCE)
  mul = (A.index((m*UOp.const(dtypes.index, K)+k))*B.index((k*UOp.const(dtypes.index, N)+n))).cast(dtypes.float32)
  red = mul.reduce(k, arg=Ops.ADD, dtype=dtypes.float32).cast(C.dtype.base)
  store = C.index((m*UOp.const(dtypes.index, N)+n), ptr=True).store(red).end(m, n)
  return store.sink(arg=KernelInfo(name=f'uop_gemm_{M}_{N}_{K}'))

# ** backward gemm, might use the asm gemm

def custom_gemm_bw(gradient:UOp, kernel:UOp):
  out, a, b = kernel.src
  assert all_same([gradient.device, a.device, b.device, out.device])
  a_t, b_t, g_t = Tensor(a, device=a.device), Tensor(b, device=a.device), Tensor(gradient, device=a.device)
  grad_a = (g_t @ b_t.T).uop
  a_T = a_t.transpose(-2, -1)
  a_T = a_T.reshape(*a_T.shape[:-1], 1, a_T.shape[-1])
  g_r = g_t.reshape(*g_t.shape[:-2], 1, *g_t.shape[-2:]).transpose(-1, -2)
  grad_b = (a_T * g_r).sum((-1, 0)).uop
  return (None, grad_a, grad_b)

# ** main gemm function

def asm_gemm(a:Tensor, b:Tensor) -> Tensor:
  assert can_use_asm_gemm(a, b), f"{counters['todos'][-1]}"
  counters["used"] += 1
  squeeze = a.ndim == 2
  if squeeze: a = a.unsqueeze(0)

  batch, M, K = a.shape
  N = b.shape[1]
  is_multi = isinstance(a.device, tuple)

  if is_multi:
    out = Tensor(Tensor.empty(batch//len(a.device), M, N, dtype=a.dtype, device=a.device).uop.multi(0), device=a.device)
  else:
    out = Tensor.empty(batch, M, N, dtype=a.dtype, device=a.device)

  dname = a.device[0] if is_multi else a.device
  arch = getattr(Device[dname].renderer, "arch", "")
  if arch.startswith("gfx950") and getenv("USE_ASM", 1):
    numWG = GEMM_ARGS[(M, N, K)][0]
    out = Tensor.custom_kernel(out, a, b, fxn=functools.partial(custom_asm_gemm, dname=dname, wg=numWG, arch=arch), grad_fxn=custom_gemm_bw)[0]
  else:
    out = Tensor.custom_kernel(out, a, b, fxn=custom_uop_gemm, grad_fxn=custom_gemm_bw)[0]
  return out.squeeze(0) if squeeze else out
