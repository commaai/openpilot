from tinygrad import Tensor, Device, GlobalCounters, TinyJit, dtypes
from tinygrad.helpers import getenv, Context, DEBUG

def test(devs: list[str], N: int, iters:int = 10, name:str = "allreduce"):
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
    print(f"{name} iter {i+1}/{iters}: {i_secs:.6f} sec {i_gflops:.2f} GFLOP/s {i_gbs:.2f} GB/s")
    secs += i_secs
    gflops += i_gflops
    gbs += i_gbs

  return (gflops/iters, gbs/iters, secs/iters)

def run(sz, n_gpus=6, iters=10, ring=0, all2all=0):
  devs = tuple([f"{Device.DEFAULT}:{x}" for x in range(n_gpus)])
  N = sz // dtypes.float32.itemsize
  name = "all2all" if all2all else ("ring" if ring else "naive")
  with Context(RING=(2 if ring else 0), ALL2ALL=(2 if all2all else 0), JIT_BATCH_SIZE=0, DEBUG=max(DEBUG.value, 2)):
    return test(devs, N, iters=iters, name=name)

def main():
  n_gpus = getenv("GPUS", 6)
  iters = getenv("ITERS", 10)
  sz = getenv("SZ", 1000) * 10**6 # size of data on each gpu
  print(f"Using {sz/10**9:.2f} GB of numbers on each of {n_gpus} GPUs, {n_gpus*sz/10**9:.2f} GB total.")

  results = {}
  for name, kwargs in [("naive", {}), ("ring", {"ring": 2}), ("all2all", {"all2all": 2})]:
    results[name] = run(sz, n_gpus=n_gpus, iters=iters, **kwargs)

  print("\n=== RESULTS ===")
  for name, (gflops, gbs, secs) in results.items():
    print(f"{name.upper()}:\n  {secs:.6f} seconds/iter\n  {gflops:.2f} GFLOP/s\n  {gbs:.2f} GB/s")

if __name__ == "__main__":
  main()
