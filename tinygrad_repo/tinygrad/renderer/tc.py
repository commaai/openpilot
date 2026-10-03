import math, functools
from dataclasses import dataclass
from tinygrad.dtype import DType, dtypes
from tinygrad.uop.ops import PatternMatcher, UOp, UPat, Ops

@dataclass(frozen=True)
class TensorCore: # D = A * B + C, A is (M x K), B is (K x N), C and D are (M x N)
  dtype_in: DType # dtype for A and B
  dtype_out: DType # dtype for C and D
  # A, B and C fragments as (lane bits, element bits), least significant bit first, in tile bits m<i>/n<i>/k<i>. C lanes are locals, elements upcasts
  # a foreign lane bit is broadcast. the k bits may be permuted, identically in A and B
  frag_a: tuple[tuple[str, ...], tuple[str, ...]]
  frag_b: tuple[tuple[str, ...], tuple[str, ...]]
  frag_c: tuple[tuple[str, ...], tuple[str, ...]]
  def axis_coords(self) -> list[str]:
    # tile bit of each tc axis in creation order, n then m then the k unrolls. split j of a dim is bit j
    used = self.frag_a[0] + self.frag_a[1] + self.frag_c[0] + self.frag_c[1]
    return [f"{d}{i}" for d in "nmk" for i in range(1+max([int(c[1:]) for c in used if c[0] == d], default=-1))]
  def relabel(self) -> list[dict[str, str]]:
    # tc axis -> fragment slot axis, per operand
    return [{c: y for y,c in zip(self.frag_c[0] + self.base_upcast_axes()[:len(f[1])][::-1], f[0]+f[1])} for f in (self.frag_a, self.frag_b)]
  @functools.cache  # pylint: disable=method-cache-max-size-none
  def frag_coords(self) -> list[list[list[tuple[int, int]]]]:
    # [operand][lane][element] -> tile coordinate
    def coord(f, ax, lane, elem):
      return tuple(sum(((v>>j)&1) << int(c[1:]) for bits,v in zip(f, (lane, elem)) for j,c in enumerate(bits) if c[0] == d) for d in ax)
    return [[[coord(f, ax, lane, elem) for elem in range(2**len(f[1]))] for lane in range(2**len(f[0]))]
            for f,ax in zip((self.frag_a, self.frag_b, self.frag_c), ("mk", "kn", "mn"))]
  @property
  def dims(self) -> tuple[int,int,int]: # N, M, K. every axis has size 2
    n, m, k = (sum(c[0] == d for c in self.axis_coords()) for d in "nmk")
    return (2**n, 2**m, 2**k)
  @property
  def threads(self) -> int: return 2**len(self.frag_c[0]) # threads that construct the warp
  def base_upcast_axes(self):
    # element slots, most significant bit first: upcast then reduce
    return (tuple(c for c in self.axis_coords() if c[0] == "k") + self.frag_c[1])[::-1]
  def __post_init__(self):
    # each own bit appears once, only lane bits may be foreign and k never is
    coords = self.axis_coords()
    for f,dims in zip((self.frag_a, self.frag_b, self.frag_c), ("mk","kn","mn")):
      own = {c for c in coords if c[0] in dims}
      assert len(f[0]) == len(self.frag_c[0]), f"fragment {f} has the wrong lane count"
      assert len(set(f[0]+f[1])) == len(f[0]+f[1]) and set(f[1]) <= own <= set(f[0]+f[1]) <= own | {c for c in coords if c[0] in "mn"}, \
        f"fragment {f} isn't distinct bits covering {dims}"
    # A and B must relabel k identically
    ka, kb = ([c for c in f[1]+f[0] if c[0] == "k"] for f in (self.frag_a, self.frag_b))
    assert ka == kb, f"{ka=} vs {kb=}"

# ***** NVIDIA *****

# https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma
def mma(K:int, di:DType, do:DType) -> TensorCore:
  # mma.m16n8kK: lane is threadID_in_group (2 k bits) then groupID (m0-m2); elements lsb first are 2**g k in a 32-bit reg, m3 (A row+8), leftover k
  k, g = [f"k{i}" for i in range(int(math.log2(K)))], int(math.log2(4//di.itemsize))
  lane, elem = tuple(k[g:g+2]), tuple(k[:g]+k[g+2:])
  # (8,16,K)
  return TensorCore(dtype_in=di, dtype_out=do, frag_a=(lane+("m0","m1","m2"), elem[:g]+("m3",)+elem[g:]),
    frag_b=(lane+("n0","n1","n2"), elem), frag_c=(("n1","n2","m0","m1","m2"), ("n0","m3")))
cuda_81616 = [mma(16,di,do) for di,do in [(dtypes.half,dtypes.float),(dtypes.bfloat16,dtypes.float),(dtypes.half,dtypes.half)]]
cuda_81632_f8 = [mma(32,di,dtypes.float) for di in [dtypes.fp8e4m3, dtypes.fp8e5m2]]
cuda_8168_f16 = [mma(8,dtypes.half,do) for do in [dtypes.float, dtypes.half]]
cuda_8168_tf32 = [mma(8,dtypes.float,dtypes.float)]
cuda_sm75: list[TensorCore] = cuda_8168_f16
cuda_sm80: list[TensorCore] = cuda_81616 + cuda_8168_f16 + cuda_8168_tf32
cuda_sm89: list[TensorCore] = cuda_sm80 + cuda_81632_f8

def get_cuda(arch): return cuda_sm89 if (ver:=int(arch[3:])) >= 89 else cuda_sm80 if ver >= 80 else cuda_sm75 if ver >= 75 else []

# ***** AMD *****

# https://gpuopen.com/learn/wmma_on_rdna3/
# (16,16,16)
amd_rdna3 = [TensorCore(dtype_in=di, dtype_out=do, frag_a=(("m0", "m1", "m2", "m3", "n0"), ("k0", "k1", "k2", "k3")),
  frag_b=(("n0", "n1", "n2", "n3", "m0"), ("k0", "k1", "k2", "k3")), frag_c=(("n0", "n1", "n2", "n3", "m0"), ("m1", "m2", "m3")))
  for di,do in [(dtypes.half,dtypes.float),(dtypes.half,dtypes.half),(dtypes.bfloat16,dtypes.float),(dtypes.int8,dtypes.int32)]]
# (16,16,16)
amd_rdna4 = [TensorCore(dtype_in=di, dtype_out=do, frag_a=(("m0", "m1", "m2", "m3", "k2"), ("k0", "k1", "k3")),
  frag_b=(("n0", "n1", "n2", "n3", "k2"), ("k0", "k1", "k3")), frag_c=(("n0", "n1", "n2", "n3", "m3"), ("m0", "m1", "m2")))
  for di,do in [(dtypes.half,dtypes.float),(dtypes.half,dtypes.half),(dtypes.bfloat16,dtypes.float),(dtypes.bfloat16,dtypes.bfloat16)]]

# https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
def mfma(K:int, di:DType, do:DType) -> TensorCore:
  # 16x16xK (7.1.4.1): A[i,k] is item k%K_L of lane i + 16*(k//K_L), K_L = K/4; fp8 K=128 is two K=64 halves, k//64 the high item (7.1.5.1)
  # NOTE: fp6 and fp4 K=128 keep K_L = 32 and would need their own case
  k, kl = [f"k{i}" for i in range(int(math.log2(K)))], int(math.log2(min(K, 64)//4))
  lane, elem = tuple(k[kl:kl+2]), tuple(k[:kl]+k[kl+2:])
  # (16,16,K)
  return TensorCore(dtype_in=di, dtype_out=do, frag_a=(("m0","m1","m2","m3")+lane, elem),
    frag_b=(("n0","n1","n2","n3")+lane, elem), frag_c=(("n0","n1","n2","n3","m2","m3"), ("m0","m1")))
amd_cdna_161616 = [mfma(16,di,dtypes.float) for di in [dtypes.half, dtypes.bfloat16]]
amd_cdna_161632 = [mfma(32,di,dtypes.float) for di in [dtypes.fp8e5m2, dtypes.fp8e4m3, dtypes.half, dtypes.bfloat16]]
amd_cdna_1616128 = [mfma(128,di,dtypes.float) for di in [dtypes.fp8e5m2, dtypes.fp8e4m3]]
amd_cdna3 = amd_cdna_161632[:2] + amd_cdna_161616
amd_cdna4 = amd_cdna_1616128 + amd_cdna_161632 + amd_cdna_161616

def get_amd(arch): return {"gfx942": amd_cdna3, "gfx950": amd_cdna4, "gfx1200": amd_rdna4, "gfx1201": amd_rdna4}.get(arch, amd_rdna3)

pm_validate_wmma_rdna3 = PatternMatcher([
  (UPat(Ops.WMMA, name="x", dtype=dtypes.int32), lambda x: x.replace(
    src=(x.src[0].bitcast(dtypes.uint32), x.src[1].bitcast(dtypes.uint32), x.src[2]))
    if x.src[0].dtype == dtypes.int8 and x.src[0].max_numel() == 16 else None),
  (UPat(Ops.WMMA, name="x", dtype=dtypes.half), lambda x: UOp(Ops.STACK, src=tuple(x.replace(
      src=(x.src[0], x.src[1], UOp(Ops.STACK, src=tuple(x.src[2].index(UOp.const(j//2, dtypes.int16))
      if j%2 == 0 else UOp.const(0.0, x.src[2].dtype)
      for j in range(x.max_numel()*2)))),
      arg=(*x.arg[:3], None)).index(UOp.const(i*2, dtypes.int16))
      for i in range(x.max_numel()))) if x.max_numel() == 8 else None),
  (UPat(Ops.WMMA, name="x"), lambda x: x.replace(
    src=(x.src[0].bitcast(dtypes.uint16), x.src[1].bitcast(dtypes.uint16), x.src[2]))
    if x.src[0].dtype == dtypes.bfloat16 and x.src[0].max_numel() == 16 else None),
])

pm_validate_wmma_rdna4 = PatternMatcher([
  (UPat(Ops.WMMA, name="x", dtype=dtypes.bfloat16), lambda x: x.replace(
    src=(x.src[0].bitcast(dtypes.uint16), x.src[1].bitcast(dtypes.uint16), x.src[2].bitcast(dtypes.uint16)))
      .bitcast(dtypes.bfloat16) if x.max_numel() == 8 and x.src[0].dtype == dtypes.bfloat16 and x.src[0].max_numel() == 8 else None),
  (UPat(Ops.WMMA, name="x", dtype=dtypes.float),
    lambda x: x.replace(src=(x.src[0].bitcast(dtypes.uint16), x.src[1].bitcast(dtypes.uint16), x.src[2]))
    if x.max_numel() == 8 and x.src[0].dtype == dtypes.bfloat16 and x.src[0].max_numel() == 8 else None)
])

pm_validate_wmma_cdna = PatternMatcher([
  (UPat(Ops.WMMA, name="x", dtype=dtypes.float),
    lambda x: x.replace(src=(x.src[0].bitcast(dtypes.uint32), x.src[1].bitcast(dtypes.uint32), x.src[2]))
    if x.arg[0][2] == 128 and x.src[0].dtype.itemsize <= 8 else None),
  (UPat(Ops.WMMA, name="x", dtype=dtypes.float),
    lambda x: x.replace(src=(x.src[0].bitcast(dtypes.uint16), x.src[1].bitcast(dtypes.uint16), x.src[2]))
    if x.max_numel() == 4 and x.src[0].dtype == dtypes.bfloat16 and x.src[0].max_numel() == 4 else None),
  (UPat(Ops.WMMA, name="x", dtype=dtypes.float),
    lambda x: x.replace(src=(x.src[0].bitcast(dtypes.uint64), x.src[1].bitcast(dtypes.uint64), x.src[2]))
    if x.max_numel() == 4 and x.src[0].dtype in dtypes.fp8_ocp and x.src[0].max_numel() == 8 else None),
])

# ***** Apple Metal *****

# (8,8,8)
metal = [TensorCore(dtype_in=di, dtype_out=do, frag_a=(("k1", "m0", "m1", "k2", "m2"), ("k0",)),
  frag_b=(("n1", "k0", "k1", "n2", "k2"), ("n0",)), frag_c=(("n1", "m0", "m1", "n2", "m2"), ("n0",)))
  for di,do in [(dtypes.float,dtypes.float),(dtypes.half,dtypes.float),
                (dtypes.half,dtypes.half),(dtypes.bfloat16,dtypes.float),(dtypes.bfloat16,dtypes.bfloat16)]]
