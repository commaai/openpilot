from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import Timing, getenv

if __name__ == "__main__":
  BS = getenv("BS", 2**14)
  BLOCKSIZE = getenv("BLOCKSIZE", 4096)
  HASHFN = getenv("HASHFN", "shake_128")
  NRUNS = getenv("NRUNS", 5)

  @TinyJit
  def hasher(data: Tensor): return data.keccak(HASHFN)

  t = Tensor.randn(BS, BLOCKSIZE, dtype=dtypes.uint8).realize()
  ds_mib = t.nbytes() / 1024**2

  print(f"--- benchmarking (hash: {HASHFN}, data size: {ds_mib} MiB, block size: {BLOCKSIZE} B, batch size: {BS})")
  for i in range(NRUNS):
    with Timing(f"run: {i+1}, elapsed time: ", (lambda et: f", throughput: {ds_mib / (et*1e-9):.2f} MiB/s")):
      hasher(t).realize()
