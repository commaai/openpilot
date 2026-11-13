import pickle, os, sys, functools, numpy as np
from pathlib import Path

os.environ["DEV"] = "CUDA"
os.environ["PROFILE"] = os.environ.get("PROFILE", "2")
from extra.nv_pma.cupti import cu_prof_ext
cu_prof_ext.enable_auto()

from tinygrad import Tensor, Device

if not os.environ.get("IOCTL") or not os.environ.get("GRAB_PMA"):
  print("Usage: GRAB_PMA=1 IOCTL=1 IOCTL_PRINT=0 python3 extra/nv_pma/collect.py")
  sys.exit(1)

assert Device.DEFAULT == "CUDA", "only works with CUDA"

EXAMPLES_DIR = Path(__file__).parent / "examples"
_collectors: list[tuple[str, callable]] = []

def pcsampling_test(name: str):
  def decorator(fn):
    @functools.wraps(fn)
    def wrapper():
      cu_prof_ext.clear_pma_raw_dumps()
      cu_prof_ext.clear_cupti_pc_samples()

      fn()
      Device["CUDA"].synchronize()

      dumps = cu_prof_ext.get_pma_raw_dumps()
      # from hexdump import hexdump
      # hexdump(dumps[0][:0x40])

      return {"test_name": name, "pma_raw_dumps": list(cu_prof_ext.get_pma_raw_dumps()), "cupti_pc_samples": list(cu_prof_ext.get_cupti_pc_samples())}
    _collectors.append((name, wrapper))
    return wrapper
  return decorator

# Refs

@pcsampling_test("test_plus")
def test_plus():
  a = Tensor([1, 2, 3, 4])
  b = Tensor([5, 6, 7, 8])
  (a + b).realize()

@pcsampling_test("test_matmul")
def test_matmul():
  a = Tensor(np.random.rand(12, 12).astype(np.float32))
  b = Tensor(np.random.rand(12, 12).astype(np.float32))
  (a @ b).realize()

@pcsampling_test("test_reduce_sum")
def test_reduce_sum():
  a = Tensor(np.random.rand(1024).astype(np.float32))
  a.sum().realize()

@pcsampling_test("test_reduce_max")
def test_reduce_max():
  a = Tensor(np.random.rand(1024).astype(np.float32))
  a.max().realize()

@pcsampling_test("test_exp")
def test_exp():
  a = Tensor(np.random.rand(256).astype(np.float32))
  a.exp().realize()

@pcsampling_test("test_softmax")
def test_softmax():
  a = Tensor(np.random.rand(64, 64).astype(np.float32))
  a.softmax().realize()

@pcsampling_test("test_conv2d")
def test_conv2d():
  x = Tensor(np.random.rand(1, 3, 32, 32).astype(np.float32))
  w = Tensor(np.random.rand(8, 3, 3, 3).astype(np.float32))
  x.conv2d(w).realize()

@pcsampling_test("test_large_matmul")
def test_large_matmul():
  a = Tensor(np.random.rand(128, 128).astype(np.float32))
  b = Tensor(np.random.rand(128, 128).astype(np.float32))
  (a @ b).realize()

@pcsampling_test("test_elementwise_chain")
def test_elementwise_chain():
  a = Tensor(np.random.rand(512).astype(np.float32))
  ((a + 1) * 2 - 0.5).relu().realize()

@pcsampling_test("test_broadcast")
def test_broadcast():
  a = Tensor(np.random.rand(64, 1).astype(np.float32))
  b = Tensor(np.random.rand(1, 64).astype(np.float32))
  (a + b).realize()

@pcsampling_test("test_plus_big")
def test_plus_big():
  a = Tensor(np.random.rand(64, 32).astype(np.float32))
  b = Tensor(np.random.rand(64, 32).astype(np.float32))
  (a + b).realize()

def save_example(name: str, data: dict):
  pma_bytes = sum(len(d) for d in data['pma_raw_dumps'])
  cupti_samples = sum(r['samples'] for r in data['cupti_pc_samples'])
  print(f"  PMA: {len(data['pma_raw_dumps'])} buffers, {pma_bytes} bytes")
  print(f"  CUPTI: {len(data['cupti_pc_samples'])} records, {cupti_samples} samples")

  outfile = EXAMPLES_DIR / f"{name}.pkl"
  with open(outfile, "wb") as f:
    pickle.dump(data, f)
  print(f"  Saved to {outfile}")

if __name__ == "__main__":
  EXAMPLES_DIR.mkdir(exist_ok=True)

  # Run specific tests if provided as arguments, otherwise run all
  if len(sys.argv) > 1:
    test_names = sys.argv[1:]
    collectors = [(name, fn) for name, fn in _collectors if name in test_names]
    if not collectors:
      print(f"Unknown tests: {test_names}")
      print(f"Available: {[name for name, _ in _collectors]}")
      sys.exit(1)
  else:
    collectors = _collectors

  for name, collect_fn in collectors:
    print(f"\nCollecting {name}...")
    try:
      data = collect_fn()
      save_example(name, data)
    except Exception as e:
      print(f"  ERROR: {e}")
      import traceback
      traceback.print_exc()
