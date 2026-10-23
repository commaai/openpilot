#!/usr/bin/env python3
"""Test that invalid instructions raise exceptions through the mock GPU stack."""
import unittest, subprocess, os, sys

class TestMockGPUInvalidInstruction(unittest.TestCase):
  def test_unsupported_instruction_raises(self):
    """Test that unsupported instructions raise immediately through the full MOCKGPU stack."""
    test_code = '''
import os, sys
from tinygrad import Tensor
from tinygrad.engine.realize import lower_and_compile, run_linear

linear = lower_and_compile((Tensor.empty(1) + 1).schedule_linear())
binary = linear.src[-1].src[0].src[3]
lib = binary.arg.replace(bytes.fromhex("0000b0bf"), bytes.fromhex("00fe017e"), 1)
try:
  run_linear(linear.substitute({binary: binary.replace(arg=lib)}, enter_calls=True))
except ValueError as error:
  print(error, file=sys.stderr, flush=True)
  os._exit(1)
'''

    env = {**os.environ, "DEV": "MOCKKFD+AMD", "HCQ_RUNTIME_DEV": "PYTHON"}
    result = subprocess.run([sys.executable, "-c", test_code], env=env, capture_output=True, text=True, timeout=9)
    self.assertEqual(result.returncode, 1)
    self.assertIn("unknown rdna3 format word=0x7e01fe00", result.stderr)

if __name__ == "__main__":
  unittest.main()
