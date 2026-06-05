#!/usr/bin/env python3
import subprocess, sys
from tinygrad.helpers import getenv

LOOPS = getenv("LOOPS", 50)
BROKEN = getenv("BROKEN", 0)
ONLY_RESET = getenv("ONLY_RESET", 0)

BROKEN_KERNEL_SCRIPT = """
from tinygrad.device import Device
from tinygrad.runtime.ops_amd import AMDProgram, AMDDevice
from tinygrad.runtime.support.compiler_amd import compile_hip
dev = Device["AMD"]
assert isinstance(dev, AMDDevice) and dev.is_am(), "Need AM driver (not KFD)"
broken_src = '''
extern "C" __attribute__((global)) void broken(int* dummy) {
  volatile int* bad_ptr = (volatile int*)0xDEAD00000000ULL;
  *bad_ptr = 0x42;
}
'''
broken_lib = compile_hip(broken_src, dev.arch)
broken_prg = AMDProgram(dev, "broken", broken_lib)
buf = dev.allocator.alloc(64)
try:
  broken_prg(buf, global_size=(1,1,1), local_size=(1,1,1), wait=True)
  print("  ERROR: Kernel did not fault!")
except RuntimeError as e:
  print(f"  Got expected error: {e}")
"""

for i in range(LOOPS):
  print(f"=== Running hive_reset.py ({i+1}/{LOOPS}) ===")
  subprocess.run([sys.executable, "extra/amdpci/hive_reset.py"], check=True)
  print("=== hive_reset complete ===")

  if BROKEN:
    print(f"=== Running broken kernel ({i+1}/{LOOPS}) ===")
    ret = subprocess.run([sys.executable, "-c", BROKEN_KERNEL_SCRIPT])
    print(f"=== broken kernel exited with code {ret.returncode} ===")
  elif not ONLY_RESET:
    print(f"=== Running test_tiny.py ({i+1}/{LOOPS}) ===")
    ret = subprocess.run([sys.executable, "test/test_tiny.py", "TestTiny.test_plus"])
    print(f"=== test_tiny.py exited with code {ret.returncode} ===")
