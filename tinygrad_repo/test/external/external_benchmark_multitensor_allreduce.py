from tinygrad import Tensor, Device, GlobalCounters, TinyJit, dtypes
from tinygrad.helpers import getenv, Context, RING, DEBUG

def test(devs: list[str], N: int, iters:int = 10):
  @TinyJit
  def f(t: Tensor) -> Tensor: t.sum(0).realize()

  secs, gflops, gbs = 0, 0, 0
  for i in range(-3, iters):
    t = Tensor.empty((len(devs), N))
    t = t.shard(devs, 0).realize()
    GlobalCounters.reset()
    f(t)
    for d in devs: Device[d].synchronize()

    if i < 0: continue # warm up jit
    i_secs = GlobalCounters.time_sum_s
    i_gflops = GlobalCounters.global_ops/i_secs/10**9
    i_gbs = (N*4)/i_secs/10**9
    print(f"{'ring_allreduce' if RING >= 2 else 'naive_allreduce'} iter {i+1}/{iters}: {i_secs:.6f} sec {i_gflops:.2f} GFLOP/s {i_gbs:.2f} GB/s")
    secs += i_secs
    gflops += i_gflops
    gbs += i_gbs

  return (gflops/iters, gbs/iters, secs/iters)

def run(sz, n_gpus=6, iters=10, use_ring=False):
  devs = tuple([f"{Device.DEFAULT}:{x}" for x in range(n_gpus)])
  N = sz // dtypes.float32.itemsize
  with Context(RING=(2 if use_ring else 0), DEBUG=max(DEBUG.value, 2)): return test(devs, N, iters=iters)

def main():
  ONLY_RING = getenv("ONLY_RING", 0)
  n_gpus = getenv("GPUS", 6)
  iters = getenv("ITERS", 10)

  if getenv("BENCHMARK_SPLIT"):
    l, r = 0, 512
    while r - l > 1:
      m = (l + r) // 2
      (ring_gflops, ring_gbs, ring_secs) = run(m * 1024 * 4, n_gpus=n_gpus, iters=100, use_ring=True)
      (naive_gflops, naive_gbs, naive_secs) = run(m * 1024 * 4, n_gpus=n_gpus, iters=100, use_ring=False)
      if ring_secs > naive_secs: l = m
      else: r = m
    print("Better split", r * 1024, "elements")
  else:
    sz = getenv("SZ", 1000) * 10**6 # size of data on each gpu
    print(f"Using {sz/10**9:.2f} GB of numbers on each of {n_gpus} GPUs, {n_gpus*sz/10**9:.2f} GB total.")
    (ring_gflops, ring_gbs, ring_secs) = run(sz, use_ring=True, n_gpus=n_gpus, iters=iters)
    if not ONLY_RING: (naive_gflops, naive_gbs, naive_secs) = run(sz, use_ring=False, n_gpus=n_gpus, iters=iters)
    print(f"Ring:\n  {ring_secs:.6f} seconds/iter\n  {ring_gflops:.2f} GFLOP/s\n  {ring_gbs:.2f} GB/s")
    if not ONLY_RING: print(f"Naive:\n  {naive_secs:.6f} seconds/iter\n  {naive_gflops:.2f} GFLOP/s\n  {naive_gbs:.2f} GB/s")

if __name__ == "__main__":
  main()
