import subprocess, sys, os, random

CHILD_SCRIPT = """
import os, random
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.runtime.ops_amd import AMDDevice

dev = Device["AMD"]
for i in range({N}):
  sz = random.randint(1, {MAX_SZ})
  data = np.random.randint(0, 256, sz, dtype=np.uint8)
  t = Tensor(data, device="AMD").contiguous().realize()
  dev.synchronize()
  result = t.numpy()
  assert (result == data).all(), f"Data mismatch at iter {{i}}"
""".strip()

def run_child(n_ops, max_sz, timeout):
  env = os.environ.copy()
  env.setdefault("SDMA_RING_SIZE", "4096")

  script = CHILD_SCRIPT.format(N=n_ops, MAX_SZ=max_sz)
  p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

  try:
    _, stderr = p.communicate(timeout=timeout)
    return ("ok" if p.returncode == 0 else "fail"), stderr.decode(errors='replace')
  except subprocess.TimeoutExpired:
    p.kill()
    p.communicate()
    return "timeout", "TIMEOUT: SDMA ring likely stuck"

if __name__ == "__main__":
  n_iters = int(os.environ.get("FUZZ_ITERS", "10000"))
  timeout = int(os.environ.get("FUZZ_TIMEOUT", "10"))
  max_sz = int(os.environ.get("FUZZ_MAX_SZ", "65536"))

  timeouts = 0
  failures = 0

  for i in range(n_iters):
    # Run child with many ops to stress the small sdma ring buffer across warm starts
    n_ops = random.randint(20, 100)
    status, stderr = run_child(n_ops=n_ops, max_sz=max_sz, timeout=timeout)
    if status == "timeout":
      timeouts += 1
      print(f"\tstderr: {stderr[:500]}")
    elif status == "fail":
      failures += 1
      print(f"\tstderr: {stderr[:500]}")
    else:
      print(f"iter {i}: ok (n_ops={n_ops})")

  print(f"\n=== Results: {n_iters} iterations, {timeouts} timeouts, {failures} failures ===")
