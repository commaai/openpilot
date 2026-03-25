import time
import threading
from collections import namedtuple
from pathlib import Path
from collections.abc import Sequence

import openpilot.system.loggerd.deleter as deleter
from openpilot.common.timeout import Timeout, TimeoutException
from openpilot.system.loggerd.tests.loggerd_tests_common import UploaderTestCase

Stats = namedtuple("Stats", ['f_bavail', 'f_blocks', 'f_frsize'])


class TestDeleter(UploaderTestCase):
  def fake_statvfs(self, d):
    return self.fake_stats

  def setup_method(self):
    self.f_type = "fcamera.hevc"
    super().setup_method()
    self.fake_stats = Stats(f_bavail=0, f_blocks=10, f_frsize=4096)
    deleter.os.statvfs = self.fake_statvfs

  def start_thread(self):
    self.end_event = threading.Event()
    self.del_thread = threading.Thread(target=deleter.deleter_thread, args=[self.end_event])
    self.del_thread.daemon = True
    self.del_thread.start()

  def join_thread(self):
    self.end_event.set()
    self.del_thread.join()

  def test_delete(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type, 1)

    self.start_thread()

    try:
      with Timeout(2, "Timeout waiting for file to be deleted"):
        while f_path.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()

  def assertDeleteOrder(self, f_paths: Sequence[Path], timeout: int = 5) -> None:
    deleted_order = []

    self.start_thread()
    try:
      with Timeout(timeout, "Timeout waiting for files to be deleted"):
        while True:
          for f in f_paths:
            if not f.exists() and f not in deleted_order:
              deleted_order.append(f)
          if len(deleted_order) == len(f_paths):
            break
          time.sleep(0.01)
    except TimeoutException:
      print("Not deleted:", [f for f in f_paths if f not in deleted_order])
      raise
    finally:
      self.join_thread()

    assert deleted_order == f_paths, "Files not deleted in expected order"

  def test_delete_order(self):
    self.assertDeleteOrder([
      self.make_file_with_data(self.seg_format.format(0), self.f_type),
      self.make_file_with_data(self.seg_format.format(1), self.f_type),
      self.make_file_with_data(self.seg_format2.format(0), self.f_type),
    ])

  def test_delete_many_preserved(self):
    self.assertDeleteOrder([
      self.make_file_with_data(self.seg_format.format(0), self.f_type),
      self.make_file_with_data(self.seg_format.format(1), self.f_type, preserve_xattr=deleter.PRESERVE_ATTR_VALUE),
      self.make_file_with_data(self.seg_format.format(2), self.f_type),
    ] + [
      self.make_file_with_data(self.seg_format2.format(i), self.f_type, preserve_xattr=deleter.PRESERVE_ATTR_VALUE)
      for i in range(5)
    ])

  def test_delete_last(self):
    self.assertDeleteOrder([
      self.make_file_with_data(self.seg_format.format(1), self.f_type),
      self.make_file_with_data(self.seg_format2.format(0), self.f_type),
      self.make_file_with_data(self.seg_format.format(0), self.f_type, preserve_xattr=deleter.PRESERVE_ATTR_VALUE),
      self.make_file_with_data("boot", self.seg_format[:-4]),
      self.make_file_with_data("crash", self.seg_format2[:-4]),
    ])

  def test_no_delete_when_available_space(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type)

    block_size = 4096
    available = (10 * 1024 * 1024 * 1024) / block_size  # 10GB free
    self.fake_stats = Stats(f_bavail=available, f_blocks=10, f_frsize=block_size)

    self.start_thread()
    start_time = time.monotonic()
    while f_path.exists() and time.monotonic() - start_time < 2:
      time.sleep(0.01)
    self.join_thread()

    assert f_path.exists(), "File deleted with available space"

  def test_no_delete_with_lock_file(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type, lock=True)

    self.start_thread()
    start_time = time.monotonic()
    while f_path.exists() and time.monotonic() - start_time < 2:
      time.sleep(0.01)
    self.join_thread()

    assert f_path.exists(), "File deleted when locked"

  def _wait_for_deletion(self, path: Path, timeout: float = 3.0) -> bool:
    """Return True if path is deleted within timeout seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      if not path.exists() and not path.is_symlink():
        return True
      time.sleep(0.01)
    return False

  def test_delete_stray_file(self):
    """Stray non-segment files in log root should be deleted when disk is low."""
    stray = Path(deleter.Paths.log_root()) / "stray_backup.tar.gz"
    stray.write_bytes(b"data")

    self.start_thread()
    deleted = self._wait_for_deletion(stray)
    self.join_thread()

    assert deleted, "Stray file was not deleted"

  def test_delete_stray_files_multiple(self):
    """Multiple stray files are all cleaned up."""
    log_root = Path(deleter.Paths.log_root())
    strays = [log_root / f"stray_{i}.bin" for i in range(3)]
    for f in strays:
      f.write_bytes(b"x" * 1024)

    self.start_thread()
    try:
      with Timeout(5, "Timeout waiting for stray files to be deleted"):
        while any(f.exists() for f in strays):
          time.sleep(0.01)
    finally:
      self.join_thread()

  def test_delete_symlink_in_log_root(self):
    """A symlink in log root is removed without following (target must survive)."""
    target = Path(deleter.Paths.log_root()) / "symlink_target_dir"
    target.mkdir()
    link = Path(deleter.Paths.log_root()) / "stray_link"
    link.symlink_to(target)

    self.start_thread()
    deleted = self._wait_for_deletion(link)
    self.join_thread()

    assert deleted, "Symlink was not deleted"
    assert target.exists(), "Symlink target was incorrectly removed"

  def test_delete_broken_symlink(self):
    """A broken (dangling) symlink in log root is removed."""
    link = Path(deleter.Paths.log_root()) / "broken_link"
    link.symlink_to("/nonexistent/path/does/not/exist")

    self.start_thread()
    deleted = self._wait_for_deletion(link)
    self.join_thread()

    assert deleted, "Broken symlink was not deleted"

  def test_stray_file_mixed_with_segments(self):
    """Stray files and segment directories coexist; both are cleaned up."""
    seg_file = self.make_file_with_data(self.seg_dir, self.f_type)
    stray = Path(deleter.Paths.log_root()) / "unexpected.log"
    stray.write_bytes(b"oops")

    self.start_thread()
    try:
      with Timeout(5, "Timeout waiting for all entries to be deleted"):
        while seg_file.exists() or stray.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()

  def test_deleter_continues_after_stray_entry_error(self):
    """A deletion failure on one stray entry does not prevent deleting the next."""
    log_root = Path(deleter.Paths.log_root())
    good_stray = log_root / "good_stray.bin"
    good_stray.write_bytes(b"deleteme")

    # Also create a normal segment so the deleter has something to process
    seg_file = self.make_file_with_data(self.seg_format.format(0), self.f_type)

    self.start_thread()
    try:
      with Timeout(5, "Timeout waiting for entries to be deleted"):
        while good_stray.exists() and seg_file.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()

  def test_delete_nested_stray_directory(self):
    """A non-segment directory nested inside log root is removed via the regular dir path."""
    # Non-segment dirs ARE included by listdir_by_creation and should be cleaned up
    non_seg_dir = Path(deleter.Paths.log_root()) / "some_random_dir"
    (non_seg_dir / "subdir").mkdir(parents=True)
    (non_seg_dir / "file.txt").write_bytes(b"hello")

    self.start_thread()
    deleted = self._wait_for_deletion(non_seg_dir)
    self.join_thread()

    assert deleted, "Non-segment directory was not deleted"
