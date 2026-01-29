import os, subprocess
from pathlib import Path
from tinygrad.helpers import temp

EXAMPLES_DIR = Path(__file__).parent
PROFILE_PATH = Path(temp("profile.pkl", append_user=True))

EXAMPLES = [
  "test.test_custom_kernel.TestCustomKernel.test_empty",
  "test.test_tiny.TestTiny.test_plus",
  "test.test_tiny.TestTiny.test_gemm",
]

if __name__ == "__main__":
  arch = subprocess.check_output(["python", "-c", "from tinygrad import Device; print(Device['AMD'].arch)"], text=True,
                                 env={**os.environ, "DEBUG":"0"}).rstrip()
  (EXAMPLES_DIR/arch).mkdir(exist_ok=True)
  for test in EXAMPLES:
    for i in range(2):
      subprocess.run(["python", "-m", "unittest", test], cwd=EXAMPLES_DIR.parent.parent.parent,
                     env={**os.environ, "AMD":"1", "SQTT_LIMIT_SE":"-1", "VIZ":"-2"}, check=True)
      PROFILE_PATH.rename(dest:=EXAMPLES_DIR/arch/f"profile_{test.split('.')[-1].replace('test_', '')}_run_{i}.pkl")
      print(f"saved SQTT trace to {dest}")
