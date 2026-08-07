import os
from uuid import uuid4

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.utils import atomic_write


class TestFileHelpers(OpenpilotTestCase):
  def run_atomic_write_func(self, atomic_write_func):
    path = f"/tmp/tmp{uuid4()}"
    with atomic_write_func(path) as f:
      f.write("test")
      assert not os.path.exists(path)

    with open(path) as f:
      assert f.read() == "test"
    os.remove(path)

  def test_atomic_write(self):
    self.run_atomic_write_func(atomic_write)

  def test_atomic_write_cleans_up_after_error(self):
    path = f"/tmp/tmp{uuid4()}"
    tmp_path = None
    try:
      with self.assertRaises(RuntimeError):
        with atomic_write(path) as f:
          tmp_path = f.name
          f.write("partial")
          raise RuntimeError("write failed")

      assert tmp_path is not None
      assert not os.path.exists(tmp_path)
      assert not os.path.exists(path)
    finally:
      for cleanup_path in (tmp_path, path):
        if cleanup_path is not None and os.path.exists(cleanup_path):
          os.remove(cleanup_path)
