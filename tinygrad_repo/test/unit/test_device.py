#!/usr/bin/env python
import unittest
from unittest.mock import patch
import os
from tinygrad import Tensor
from tinygrad.device import Device, Compiler
from tinygrad.helpers import diskcache_get, diskcache_put, getenv

class TestDevice(unittest.TestCase):
  def test_canonicalize(self):
    self.assertEqual(Device.canonicalize(None), Device.DEFAULT)
    self.assertEqual(Device.canonicalize("CPU"), "CPU")
    self.assertEqual(Device.canonicalize("cpu"), "CPU")
    self.assertEqual(Device.canonicalize("GPU"), "GPU")
    self.assertEqual(Device.canonicalize("GPU:0"), "GPU")
    self.assertEqual(Device.canonicalize("gpu:0"), "GPU")
    self.assertEqual(Device.canonicalize("GPU:1"), "GPU:1")
    self.assertEqual(Device.canonicalize("gpu:1"), "GPU:1")
    self.assertEqual(Device.canonicalize("GPU:2"), "GPU:2")
    self.assertEqual(Device.canonicalize("disk:/dev/shm/test"), "DISK:/dev/shm/test")
    self.assertEqual(Device.canonicalize("disk:000.txt"), "DISK:000.txt")

  def test_getitem_not_exist(self):
    with self.assertRaises(ModuleNotFoundError):
      Device["TYPO"]

class MockCompiler(Compiler):
  def __init__(self, key): super().__init__(key)
  def compile(self, src) -> bytes: return src.encode()

class TestCompiler(unittest.TestCase):
  def test_compile_cached(self):
    diskcache_put("key", "123", None) # clear cache
    getenv.cache_clear()
    with patch.dict(os.environ, {"DISABLE_COMPILER_CACHE": "0"}, clear=True):
      self.assertEqual(MockCompiler("key").compile_cached("123"), str.encode("123"))
      self.assertEqual(diskcache_get("key", "123"), str.encode("123"))

  def test_compile_cached_disabled(self):
    diskcache_put("disabled_key", "123", None) # clear cache
    getenv.cache_clear()
    with patch.dict(os.environ, {"DISABLE_COMPILER_CACHE": "1"}, clear=True):
      self.assertEqual(MockCompiler("disabled_key").compile_cached("123"), str.encode("123"))
      self.assertIsNone(diskcache_get("disabled_key", "123"))

  def test_device_compile(self):
    getenv.cache_clear()
    with patch.dict(os.environ, {"DISABLE_COMPILER_CACHE": "1"}):
      a = Tensor([0.,1.], device=Device.DEFAULT).realize()
      (a + 1).realize()

if __name__ == "__main__":
  unittest.main()
