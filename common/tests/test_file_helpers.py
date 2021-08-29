import os
import unittest
from uuid import uuid4

from common.file_helpers import atomic_write_on_fs_tmp
from common.file_helpers import atomic_write_in_dir


class TestFileHelpers(unittest.TestCase):
  def run_atomic_write_func(self, atomic_write_func):
    path = "/tmp/tmp{}".format(uuid4())
    with atomic_write_func(path) as f:
      f.write("test")

    with open(path) as f:
      self.assertEqual(f.read(), "test")
    os.remove(path)

  def test_atomic_write_on_fs_tmp(self):
    self.run_atomic_write_func(atomic_write_on_fs_tmp)

  def test_atomic_write_in_dir(self):
    self.run_atomic_write_func(atomic_write_in_dir)


if __name__ == "__main__":
  unittest.main()
