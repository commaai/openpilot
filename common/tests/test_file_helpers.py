import os
from uuid import uuid4

from openpilot.common.file_helpers import atomic_write_in_dir


class TestFileHelpers:
  def run_atomic_write_func(self, atomic_write_func):
    path = f"/tmp/tmp{uuid4()}"
    with atomic_write_func(path) as f:
      f.write("test")
      assert not os.path.exists(path)

    with open(path) as f:
      assert f.read() == "test"
    os.remove(path)

  def test_atomic_write_in_dir(self):
    self.run_atomic_write_func(atomic_write_in_dir)
