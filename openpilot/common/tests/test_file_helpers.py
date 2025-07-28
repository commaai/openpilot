from os import path, remove
from uuid import uuid4

from openpilot.common.file_helpers import atomic_write_in_dir


class TestFileHelpers:
    def run_atomic_write_func(self, atomic_write_func):
        filepath = f"/tmp/tmp{uuid4()}"
        with atomic_write_func(filepath) as f:
            f.write("test")
            assert not path.exists(filepath)

        with open(filepath) as f:
            assert f.read() == "test"
        remove(filepath)

    def test_atomic_write_in_dir(self):
        self.run_atomic_write_func(atomic_write_in_dir)
