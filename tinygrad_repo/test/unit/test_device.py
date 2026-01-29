#!/usr/bin/env python
import unittest, os, subprocess
from tinygrad import Tensor
from tinygrad.device import Device, Compiler, enumerate_devices_str
from tinygrad.helpers import diskcache_get, diskcache_put, getenv, Context, WIN, CI

class TestDevice(unittest.TestCase):
  def test_canonicalize(self):
    self.assertEqual(Device.canonicalize(None), Device.DEFAULT)
    self.assertEqual(Device.canonicalize("CPU"), "CPU")
    self.assertEqual(Device.canonicalize("cpu"), "CPU")
    self.assertEqual(Device.canonicalize("CL"), "CL")
    self.assertEqual(Device.canonicalize("CL:0"), "CL")
    self.assertEqual(Device.canonicalize("cl:0"), "CL")
    self.assertEqual(Device.canonicalize("CL:1"), "CL:1")
    self.assertEqual(Device.canonicalize("cl:1"), "CL:1")
    self.assertEqual(Device.canonicalize("CL:2"), "CL:2")
    self.assertEqual(Device.canonicalize("disk:/dev/shm/test"), "DISK:/dev/shm/test")
    self.assertEqual(Device.canonicalize("disk:000.txt"), "DISK:000.txt")

  def test_getitem_not_exist(self):
    with self.assertRaises(ModuleNotFoundError):
      Device["TYPO"]

  def test_lowercase_canonicalizes(self):
    device = Device.DEFAULT
    Device.DEFAULT = device.lower()
    self.assertEqual(Device.canonicalize(None), device)
    Device.DEFAULT = device

  @unittest.skipIf(WIN and CI, "skipping windows test") # TODO: subprocess causes memory violation?
  def test_env_overwrite_default_compiler(self):
    if Device.DEFAULT == "CPU":
      from tinygrad.runtime.support.compiler_cpu import CPULLVMCompiler, ClangJITCompiler
      try: _, _ = CPULLVMCompiler(), ClangJITCompiler()
      except Exception as e: self.skipTest(f"skipping compiler test: not all compilers: {e}")

      imports = "from tinygrad import Device; from tinygrad.runtime.support.compiler_cpu import CPULLVMCompiler, ClangJITCompiler"
      subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, CPULLVMCompiler)"'],
                        shell=True, check=True, env={**os.environ, "DEV": "CPU", "CPU_LLVM": "1"})
      subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, ClangJITCompiler)"'],
                        shell=True, check=True, env={**os.environ, "DEV": "CPU", "CPU_LLVM": "0"})
      subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, CPULLVMCompiler)"'],
                        shell=True, check=True, env={**os.environ, "DEV": "CPU", "CPU_CC": "LLVM"})
      subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, ClangJITCompiler)"'],
                        shell=True, check=True, env={**os.environ, "DEV": "CPU", "CPU_CC": "CLANGJIT"})
    elif Device.DEFAULT == "AMD":
      from tinygrad.runtime.support.compiler_amd import HIPCompiler, AMDLLVMCompiler
      try: _, _ = HIPCompiler(Device[Device.DEFAULT].arch), AMDLLVMCompiler(Device[Device.DEFAULT].arch)
      except Exception as e: self.skipTest(f"skipping compiler test: not all compilers: {e}")

      imports = "from tinygrad import Device; from tinygrad.runtime.support.compiler_amd import HIPCompiler, AMDLLVMCompiler"
      subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, AMDLLVMCompiler)"'],
                        shell=True, check=True, env={**os.environ, "DEV": "AMD", "AMD_LLVM": "1"})
      subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, HIPCompiler)"'],
                        shell=True, check=True, env={**os.environ, "DEV": "AMD", "AMD_LLVM": "0"})
      subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, AMDLLVMCompiler)"'],
                        shell=True, check=True, env={**os.environ, "DEV": "AMD", "AMD_CC": "LLVM"})
      subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, HIPCompiler)"'],
                        shell=True, check=True, env={**os.environ, "DEV": "AMD", "AMD_CC": "HIP"})
    else: self.skipTest("only run on CPU/AMD")

  @unittest.skipIf((WIN and CI) or (not Device.DEFAULT == "CPU"), "skipping windows test")
  def test_env_online(self):
    from tinygrad.runtime.support.compiler_cpu import CPULLVMCompiler, ClangJITCompiler
    try: _, _ = CPULLVMCompiler(), ClangJITCompiler()
    except Exception as e: self.skipTest(f"skipping compiler test: not all compilers: {e}")

    with Context(CPU_LLVM=1):
      inst = Device["CPU"].compiler
      self.assertIsInstance(Device["CPU"].compiler, CPULLVMCompiler)
    with Context(CPU_LLVM=0):
      self.assertIsInstance(Device["CPU"].compiler, ClangJITCompiler)
    with Context(CPU_LLVM=1):
      self.assertIsInstance(Device["CPU"].compiler, CPULLVMCompiler)
      assert inst is Device["CPU"].compiler  # cached

class MockCompiler(Compiler):
  def __init__(self, key): super().__init__(key)
  def compile(self, src) -> bytes: return src.encode()

class TestCompiler(unittest.TestCase):
  def test_compile_cached(self):
    diskcache_put("key", "123", None) # clear cache
    getenv.cache_clear()
    with Context(CCACHE=1):
      self.assertEqual(MockCompiler("key").compile_cached("123"), str.encode("123"))
      self.assertEqual(diskcache_get("key", "123"), str.encode("123"))

  def test_compile_cached_disabled(self):
    diskcache_put("disabled_key", "123", None) # clear cache
    getenv.cache_clear()
    with Context(CCACHE=0):
      self.assertEqual(MockCompiler("disabled_key").compile_cached("123"), str.encode("123"))
      self.assertIsNone(diskcache_get("disabled_key", "123"))

  def test_device_compile(self):
    getenv.cache_clear()
    with Context(CCACHE=0):
      a = Tensor([0.,1.], device=Device.DEFAULT).realize()
      (a + 1).realize()

class TestRunAsModule(unittest.TestCase):
  def test_module_runs(self):
    out = '\n'.join(enumerate_devices_str())
    self.assertIn("CPU", out) # for sanity check

if __name__ == "__main__":
  unittest.main()
