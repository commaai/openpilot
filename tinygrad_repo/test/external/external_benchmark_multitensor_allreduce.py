import time
from tinygrad import Tensor, Device, GlobalCounters, TinyJit
from tinygrad.ops import Ops, UOp
from tinygrad.multi import MultiLazyBuffer, all_reduce
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import run_schedule
from tinygrad.helpers import getenv, Context, RING, DEBUG
from typing import List, Union

def realize(x: Union[UOp, List[UOp]]):
  x = x if isinstance(x, list) else [x]
  run_schedule(create_schedule(x))
  for lb in x: Device[lb.device].synchronize()

def test(devs: List[str], N: int, iters:int = 10):
  def _wrapped(op: Ops, t: Tensor) -> Tensor:
    return Tensor(MultiLazyBuffer(all_reduce(op, t.lazydata.lbs), 0), device=devs)
  _jitted = TinyJit(_wrapped) if getenv("USEJIT", 1) == 1 else _wrapped

  secs, gflops, gbs = 0, 0, 0
  for i in range(-2, iters):
    lbs = [Tensor.full((N,), float(1+i), device=d).contiguous().lazydata for i,d in enumerate(devs)]
    realize(lbs)
    GlobalCounters.reset()
    start = time.time()
    realize(_jitted(Ops.ADD, Tensor(MultiLazyBuffer(lbs, 0), device=devs)).lazydata.lbs)
    if i < 0: continue # warm up jit
    i_secs = time.time() - start

    if DEBUG >= 2: i_secs = GlobalCounters.time_sum_s
    i_gflops = GlobalCounters.global_ops/i_secs/10**9
    i_gbs = (N*4)/i_secs/10**9
    print(f"{'ring_allreduce' if RING >= 2 else 'naive_allreduce'} iter {i+1}/{iters}: {i_secs:.6f} sec {i_gflops:.2f} GFLOP/s {i_gbs:.2f} GB/s")
    secs += i_secs
    gflops += i_gflops
    gbs += i_gbs

  return (gflops/iters, gbs/iters, secs/iters)

def run(sz, n_gpus=6, iters=10):
  dev = Device.DEFAULT
  devs = tuple([f"{dev}:{x}" for x in range(n_gpus)])
  N = sz // 4 # float32 is 4 bytes

  with Context(RING=2):
    (ring_gflops, ring_gbs, ring_secs) = test(devs, N, iters=iters)
  with Context(RING=0):
    (naive_gflops, naive_gbs, naive_secs) = test(devs, N, iters=iters)
  return (ring_gflops, ring_gbs, ring_secs), (naive_gflops, naive_gbs, naive_secs)

def main():
  n_gpus = getenv("GPUS", 6)

  if getenv("BENCHMARK_SPLIT"):
    l, r = 0, 512
    while r - l > 1:
      m = (l + r) // 2
      (ring_gflops, ring_gbs, ring_secs), (naive_gflops, naive_gbs, naive_secs) = run(m * 1024 * 4, n_gpus=n_gpus, iters=100)
      if ring_secs > naive_secs: l = m
      else: r = m
    print("Better split", r * 1024, "elements")
  else:
    sz = getenv("SZ", 1000) * 10**6 # size of data on each gpu
    print(f"Using {sz/10**9:.2f} GB of numbers on each of {n_gpus} GPUs, {n_gpus*sz/10**9:.2f} GB total.")
    (ring_gflops, ring_gbs, ring_secs), (naive_gflops, naive_gbs, naive_secs) = run(sz)
    print(f"Ring:\n  {ring_secs:.6f} seconds/iter\n  {ring_gflops:.2f} GFLOP/s\n  {ring_gbs:.2f} GB/s")
    print(f"Naive:\n  {naive_secs:.6f} seconds/iter\n  {naive_gflops:.2f} GFLOP/s\n  {naive_gbs:.2f} GB/s")

if __name__ == "__main__":
  main()
