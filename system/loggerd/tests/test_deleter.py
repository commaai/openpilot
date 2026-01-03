import os
import time
import threading
from collections import namedtuple
from pathlib import Path
from collections.abc import Sequence

import openpilot.system.loggerd.deleter as deleter
from openpilot.common.timeout import Timeout, TimeoutException
from openpilot.system.loggerd.tests.loggerd_tests_common import UploaderTestCase
from openpilot.system.hardware.hw import Paths

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

  def test_delete_stray_file_in_log_root(self):
    """Test deletion of a stray file directly in log root."""
    log_root = Path(Paths.log_root())
    stray_file = log_root / "stray_file.txt"
    stray_file.write_text("stray content")

    self.start_thread()
    try:
      with Timeout(2, "Timeout waiting for stray file to be deleted"):
        while stray_file.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()

    assert not stray_file.exists(), "Stray file not deleted"

  def test_delete_symlink_in_log_root(self):
    """Test deletion of a symlink directly in log root."""
    log_root = Path(Paths.log_root())

    # Create a target file outside log root
    target_file = log_root / "target.txt"
    target_file.write_text("target content")

    # Create a symlink in log root
    symlink = log_root / "symlink_to_target"
    symlink.symlink_to(target_file)

    self.start_thread()
    try:
      with Timeout(2, "Timeout waiting for symlink to be deleted"):
        while symlink.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()

    assert not symlink.exists(), "Symlink not deleted"
    assert target_file.exists(), "Target file was deleted (should only delete symlink)"

    # Clean up target file
    target_file.unlink()

  def test_delete_nested_directory_structure(self):
    """Test deletion of deeply nested directory structures."""
    log_root = Path(Paths.log_root())
    nested_dir = log_root / "nested" / "deep" / "structure"
    nested_dir.mkdir(parents=True, exist_ok=True)

    # Create files at various levels
    (log_root / "nested" / "file1.txt").write_text("content1")
    (log_root / "nested" / "deep" / "file2.txt").write_text("content2")
    (nested_dir / "file3.txt").write_text("content3")

    self.start_thread()
    try:
      with Timeout(2, "Timeout waiting for nested directory to be deleted"):
        while (log_root / "nested").exists():
          time.sleep(0.01)
    finally:
      self.join_thread()

    assert not (log_root / "nested").exists(), "Nested directory not deleted"

  def test_delete_mixed_items_in_log_root(self):
    """Test deletion of various types of items: files, symlinks, directories."""
    log_root = Path(Paths.log_root())

    # Create various items
    items = []

    # Regular file
    regular_file = log_root / "regular.txt"
    regular_file.write_text("regular content")
    items.append(regular_file)

    # Empty directory
    empty_dir = log_root / "empty_dir"
    empty_dir.mkdir(exist_ok=True)
    items.append(empty_dir)

    # Directory with files
    dir_with_files = log_root / "dir_with_files"
    dir_with_files.mkdir(exist_ok=True)
    (dir_with_files / "file.txt").write_text("content")
    items.append(dir_with_files)

    # Symlink to file
    target = log_root / "target.txt"
    target.write_text("target")
    symlink = log_root / "link_to_file"
    symlink.symlink_to(target)
    items.append(symlink)

    self.start_thread()
    try:
      with Timeout(5, "Timeout waiting for mixed items to be deleted"):
        while any(item.exists() for item in items if item != target):
          time.sleep(0.01)
    finally:
      self.join_thread()

    # All items except target should be deleted
    for item in items:
      if item != target:
        assert not item.exists(), f"{item} not deleted"

    # Clean up target
    if target.exists():
      target.unlink()

  def test_delete_with_permission_errors(self):
    """Test that deleter continues after encountering permission errors."""
    # Create two directories
    dir1_path = self.make_file_with_data(self.seg_format.format(0), self.f_type)
    dir2_path = self.make_file_with_data(self.seg_format.format(1), self.f_type)

    # Make dir1 read-only (simulate permission error)
    dir1 = dir1_path.parent
    try:
      os.chmod(dir1, 0o444)

      self.start_thread()
      try:
        # dir2 should still be deleted even if dir1 fails
        with Timeout(3, "Timeout waiting for dir2 to be deleted"):
          while dir2_path.exists():
            time.sleep(0.01)
      finally:
        self.join_thread()

      # Restore permissions for cleanup
      os.chmod(dir1, 0o755)

    except (OSError, PermissionError):
      # On some systems we can't change permissions, skip this test
      os.chmod(dir1, 0o755)
      self.join_thread()
      return

  def test_delete_special_characters_in_names(self):
    """Test deletion of files/directories with special characters in names."""
    log_root = Path(Paths.log_root())

    # Create items with special characters
    special_names = [
      "file with spaces.txt",
      "file-with-dashes.txt",
      "file_with_underscores.txt",
      "file.multiple.dots.txt",
    ]

    items = []
    for name in special_names:
      item = log_root / name
      item.write_text("content")
      items.append(item)

    self.start_thread()
    try:
      with Timeout(3, "Timeout waiting for special character files to be deleted"):
        while any(item.exists() for item in items):
          time.sleep(0.01)
    finally:
      self.join_thread()

    for item in items:
      assert not item.exists(), f"{item} not deleted"

  def test_delete_empty_files_and_directories(self):
    """Test deletion of empty files and directories."""
    log_root = Path(Paths.log_root())

    # Create empty file
    empty_file = log_root / "empty.txt"
    empty_file.touch()

    # Create empty directory
    empty_dir = log_root / "empty_dir"
    empty_dir.mkdir(exist_ok=True)

    self.start_thread()
    try:
      with Timeout(2, "Timeout waiting for empty items to be deleted"):
        while empty_file.exists() or empty_dir.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()

    assert not empty_file.exists(), "Empty file not deleted"
    assert not empty_dir.exists(), "Empty directory not deleted"
