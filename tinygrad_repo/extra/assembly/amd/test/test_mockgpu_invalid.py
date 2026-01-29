#!/usr/bin/env python3
"""Test that invalid instructions raise exceptions through the mock GPU stack."""
import unittest, subprocess, os, time

class TestMockGPUInvalidInstruction(unittest.TestCase):
  def test_unsupported_instruction_raises(self):
    """Test that unsupported instructions raise immediately through the full MOCKGPU stack."""
    test_code = '''
import struct
from tinygrad import Device, Tensor
from tinygrad.engine.realize import get_runner
from tinygrad.runtime.ops_amd import AMDProgram

dev = Device["AMD"]
a = Tensor([1.0]).realize()
b = a + 1
si = b.schedule()[-1]
runner = get_runner(dev.device, si.ast)

prg = runner._prg
lib = bytearray(prg.lib)

# Find s_endpgm (0xBFB00000) and replace with V_MOVRELD_B32 (op=66) which has no pcode
# VOP1 encoding: bits[31:25]=0x7E, op=bits[16:9], so op=66 -> 66<<9 = 0x8400
found = False
for i in range(0, len(lib) - 4, 4):
  if struct.unpack("<I", lib[i:i+4])[0] == 0xBFB00000:
    lib[i:i+4] = struct.pack("<I", 0x7E008400)
    found = True
    break
assert found, "s_endpgm not found"

patched_prg = AMDProgram(dev, "patched", bytes(lib))
b.uop.buffer.allocate()
patched_prg(b.uop.buffer._buf, a.uop.buffer._buf, global_size=(1,1,1), local_size=(1,1,1))
dev.synchronize()
'''

    env = os.environ.copy()
    env["AMD"] = "1"
    env["MOCKGPU"] = "1"
    env["PYTHON_REMU"] = "1"
    env["HCQDEV_WAIT_TIMEOUT_MS"] = "10000"

    st = time.perf_counter()
    result = subprocess.run(["python", "-c", test_code], env=env, capture_output=True, text=True, timeout=60)
    elapsed = time.perf_counter() - st

    self.assertNotEqual(result.returncode, 0, "should have raised")
    self.assertTrue("Error" in result.stderr, f"expected an error in stderr, got: {result.stderr[:500]}")
    # Should exit immediately, not wait for the full timeout
    self.assertLess(elapsed, 9.0, f"should exit immediately on emulator exception, took {elapsed:.1f}s")

if __name__ == "__main__":
  unittest.main()
