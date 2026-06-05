import os, subprocess, sys, shlex
from pathlib import Path
from tinygrad.helpers import temp, getenv

EXAMPLES_DIR = Path(__file__).parent
PROFILE_PATH = Path(temp("profile.pkl", append_user=True))

EXAMPLES = {
  "empty":"test/backend/test_custom_kernel.py TestCustomKernel.test_empty",
  "plus":"test/test_tiny.py TestTiny.test_plus",
  "gemm":"-c \"from tinygrad import Tensor; (Tensor.empty(N:=32, N)@Tensor.empty(N, N)).realize()\"",
  "sync":"test/amd/test_custom_kernel.py TestCustomKernel.test_lds_sync",
  "handwritten":"test/amd/test_custom_kernel.py TestCustomKernel.test_handwritten",
}

if __name__ == "__main__":
  arch = subprocess.check_output(["python", "-c", "from tinygrad import Device; print(Device['AMD'].arch)"], text=True,
                                 env={**os.environ, "DEBUG":"0"}).rstrip()
  (EXAMPLES_DIR/arch).mkdir(exist_ok=True)
  for name,test in EXAMPLES.items():
    if getenv("NAME", name) != name: continue
    for i in range(2):
      # AM_RESET=1 gets a clear trace, does not work on mi300 machines
      subprocess.run([sys.executable, *shlex.split(test)], cwd=EXAMPLES_DIR.parent.parent.parent,
                     env={**os.environ, "DEV":"AMD", "AM_RESET":"1" if not arch.startswith("gfx9") else "0", "VIZ":"-2", "PYTHONPATH":"."})
      PROFILE_PATH.rename(dest:=EXAMPLES_DIR/arch/f"profile_{name}_run_{i}.pkl")
      print(f"saved SQTT trace to {dest}")
