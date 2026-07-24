from tinygrad import Tensor, dtypes, Device, TinyJit
from tinygrad.helpers import getenv, Timing, round_up

# llama 70b shapes
DIM, HIDDEN, N_HEADS, N_KV_HEADS, HEAD_DIM = 8192, 28672, 64, 8, 128
QKV_OUT = N_HEADS*HEAD_DIM + 2*N_KV_HEADS*HEAD_DIM
LAYER_SHAPES = {
  "wqkv": ((QKV_OUT, DIM), 0),
  "wo": ((DIM, N_HEADS*HEAD_DIM), 1),
  "w1": ((HIDDEN, DIM), 0),
  "w3": ((HIDDEN, DIM), 0),
  "w2": ((DIM, HIDDEN), 1),
}

def build_params(layers:int, devs:tuple[str,...], dtype):
  ts, nbytes = [], 0
  ndev = len(devs)
  for _ in range(layers):
    for (a,b), ax in LAYER_SHAPES.values():
      shape = (round_up(a, ndev) if ax==0 else a, round_up(b, ndev) if ax==1 else b)
      t = Tensor.empty(*shape, dtype=dtype).shard(devs, axis=ax).realize()
      ts.append(t)
      nbytes += t.numel()*dtype.itemsize
  return ts, nbytes

def sync():
  Device["CPU"].synchronize()
  Device[Device.DEFAULT].synchronize()

if __name__ == "__main__":
  ngpus = getenv("GPUS", 8)
  layers = getenv("LAYERS", 8)
  devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(ngpus))

  # to cpu
  grads, nbytes = build_params(layers, devs, dtypes.bfloat16)
  cpu_dst = [Tensor.rand(*g.shape, dtype=dtypes.bfloat16, device="CPU").contiguous().realize() for g in grads]
  sync()
  def to_cpu():
    for d,g in zip(cpu_dst, grads):
      d.assign(g.to("CPU"))
    return Tensor.realize(*cpu_dst)
  to_cpu_jit = TinyJit(to_cpu)
  for _ in range(3):
    to_cpu_jit()
    sync()
  for _ in range(5):
    with Timing("to cpu ", on_exit=lambda ns: f" @ {nbytes/ns:.2f} GB/s"):
      to_cpu_jit()
      sync()

  # to gpu
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE
  w_gpu, nbytes = build_params(layers, devs, FP8_DTYPE)
  cpu_src = [Tensor.rand(*w.shape, dtype=FP8_DTYPE, device="CPU").contiguous().realize() for w in w_gpu]
  sync()
  def to_gpu():
    for w,c in zip(w_gpu, cpu_src):
      w.assign(c.shard_like(w))
    return Tensor.realize(*w_gpu)
  to_gpu_jit = TinyJit(to_gpu)
  for _ in range(3):
    to_gpu_jit()
    sync()
  for _ in range(5):
    with Timing("to gpu ", on_exit=lambda ns: f" @ {nbytes/ns:.2f} GB/s"):
      to_gpu_jit()
      sync()
