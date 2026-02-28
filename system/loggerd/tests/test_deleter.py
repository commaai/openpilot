import os
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

  # -- robustness tests for non-standard entries in the log root --

  def _wait_for_path_removal(self, path: Path, timeout: float = 3) -> bool:
    """Wait for a path to be removed. Returns True if removed.
    Uses os.path.lexists to correctly detect broken symlinks."""
    self.start_thread()
    try:
      start = time.monotonic()
      while os.path.lexists(path) and time.monotonic() - start < timeout:
        time.sleep(0.01)
      return not os.path.lexists(path)
    finally:
      self.join_thread()

  def test_delete_stray_file(self):
    """A regular file (e.g. tar.gz archive) in the log root should be deleted."""
    from openpilot.system.hardware.hw import Paths
    stray = Path(Paths.log_root()) / "2023-09-23.tar.gz"
    stray.write_bytes(os.urandom(1024))

    assert self._wait_for_path_removal(stray), "Stray file was not deleted"

  def test_delete_broken_symlink(self):
    """A symlink pointing to a non-existent target should be deleted."""
    from openpilot.system.hardware.hw import Paths
    link = Path(Paths.log_root()) / "broken-link"
    link.symlink_to("/does/not/exist")

    assert self._wait_for_path_removal(link), "Broken symlink was not deleted"

  def test_delete_valid_symlink_to_file(self):
    """A symlink pointing to a real file should be removed (unlinked)."""
    from openpilot.system.hardware.hw import Paths
    target = Path(Paths.log_root()) / "target.txt"
    target.write_bytes(b"data")
    link = Path(Paths.log_root()) / "link-to-file"
    link.symlink_to(target)

    assert self._wait_for_path_removal(link), "Valid file symlink was not deleted"

  def test_delete_empty_file(self):
    """A zero-byte file should be deleted."""
    from openpilot.system.hardware.hw import Paths
    empty = Path(Paths.log_root()) / "empty_file"
    empty.touch()

    assert self._wait_for_path_removal(empty), "Empty file was not deleted"

  def test_delete_hidden_file(self):
    """A dotfile should be deleted."""
    from openpilot.system.hardware.hw import Paths
    hidden = Path(Paths.log_root()) / ".hidden_temp"
    hidden.write_bytes(b"hidden")

    assert self._wait_for_path_removal(hidden), "Hidden file was not deleted"

  def test_delete_fifo(self):
    """A named pipe (FIFO) should be deleted."""
    from openpilot.system.hardware.hw import Paths
    fifo = Path(Paths.log_root()) / "test.fifo"
    os.mkfifo(fifo)

    assert self._wait_for_path_removal(fifo), "FIFO was not deleted"

  def test_stray_files_deleted_before_segments(self):
    """Non-directory entries should be cleaned up alongside segment dirs."""
    from openpilot.system.hardware.hw import Paths

    stray = Path(Paths.log_root()) / "stray_backup.tar.gz"
    stray.write_bytes(os.urandom(512))

    seg_path = self.make_file_with_data(self.seg_dir, self.f_type)

    self.start_thread()
    try:
      with Timeout(3, "Timeout waiting for cleanup"):
        while stray.exists() or seg_path.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()

  def test_mixed_entries_no_crash(self):
    """Deleter should handle a mix of normal segments, stray files, symlinks,
    and empty directories without crashing."""
    from openpilot.system.hardware.hw import Paths
    root = Path(Paths.log_root())

    # normal segment directory
    seg_path = self.make_file_with_data(self.seg_format.format(0), self.f_type)

    # stray regular files
    (root / "backup.tar.gz").write_bytes(os.urandom(256))
    (root / "notes.txt").write_bytes(b"test")
    (root / ".DS_Store").write_bytes(b"\x00" * 32)

    # broken symlink
    (root / "dead-link").symlink_to("/nonexistent/path")

    # empty directory (not a segment)
    (root / "tmp_work").mkdir()

    # FIFO
    os.mkfifo(root / "pipe")

    self.start_thread()
    try:
      with Timeout(5, "Timeout waiting for mixed cleanup"):
        # wait for all non-directory entries to be cleaned up
        while any(os.path.lexists(root / f) for f in ["backup.tar.gz", "notes.txt", ".DS_Store", "dead-link", "pipe"]):
          time.sleep(0.01)
        # segment should also be deleted eventually
        while seg_path.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()
