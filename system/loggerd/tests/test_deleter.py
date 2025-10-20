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

  def test_delete_non_openpilot_file_in_root(self):
    """Test that non-openpilot files directly in log_root are deleted"""
    # Create a regular file directly in log_root (not in a directory)
    log_root = Path(deleter.Paths.log_root())
    non_op_file = log_root / "2023-09-23.tar.gz"
    non_op_file.write_bytes(b"x" * 1024 * 1024)  # 1MB file

    # Also create a normal openpilot directory to be deleted first
    normal_dir = self.make_file_with_data(self.seg_dir, self.f_type, 0.1)

    self.start_thread()
    try:
      with Timeout(3, "Timeout waiting for files to be deleted"):
        while normal_dir.exists() or non_op_file.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()

  def test_delete_non_openpilot_directory(self):
    """Test that non-openpilot directories are handled"""
    log_root = Path(deleter.Paths.log_root())
    non_op_dir = log_root / "realdata"
    non_op_dir.mkdir(parents=True, exist_ok=True)
    (non_op_dir / "somefile.txt").write_text("test data")
    (non_op_dir / "subdir").mkdir(exist_ok=True)
    (non_op_dir / "subdir" / "nested.txt").write_text("nested")

    self.start_thread()
    try:
      with Timeout(3, "Timeout waiting for directory to be deleted"):
        while non_op_dir.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()

  def test_delete_symlink_to_file(self):
    """Test that symlinks to files are handled"""
    log_root = Path(deleter.Paths.log_root())

    # Create a target file outside log_root
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
      tmp.write(b"target file")
      target_file = Path(tmp.name)

    try:
      # Create symlink in log_root pointing to external file
      symlink_path = log_root / "external_link.txt"
      symlink_path.symlink_to(target_file)

      self.start_thread()
      try:
        with Timeout(3, "Timeout waiting for symlink to be deleted"):
          while symlink_path.exists():
            time.sleep(0.01)
      finally:
        self.join_thread()

      # Target file should still exist
      assert target_file.exists(), "Target file was deleted (should only delete symlink)"
    finally:
      # Cleanup
      if target_file.exists():
        target_file.unlink()

  def test_delete_symlink_to_directory(self):
    """Test that symlinks to directories are handled"""
    log_root = Path(deleter.Paths.log_root())

    # Create a target directory with files
    import tempfile
    target_dir = Path(tempfile.mkdtemp())
    (target_dir / "file.txt").write_text("data")

    try:
      # Create symlink in log_root pointing to external directory
      symlink_path = log_root / "external_dir_link"
      symlink_path.symlink_to(target_dir)

      self.start_thread()
      try:
        with Timeout(3, "Timeout waiting for symlink to be deleted"):
          while symlink_path.exists():
            time.sleep(0.01)
      finally:
        self.join_thread()

      # Target directory should still exist
      assert target_dir.exists(), "Target directory was deleted (should only delete symlink)"
      assert (target_dir / "file.txt").exists(), "Target directory contents were deleted"
    finally:
      # Cleanup
      if target_dir.exists():
        import shutil
        shutil.rmtree(target_dir)

  def test_delete_broken_symlink(self):
    """Test that broken symlinks are handled"""
    log_root = Path(deleter.Paths.log_root())

    # Create a symlink to non-existent target
    broken_link = log_root / "broken_link"
    broken_link.symlink_to("/nonexistent/path/to/nowhere")

    self.start_thread()
    try:
      with Timeout(3, "Timeout waiting for broken symlink to be deleted"):
        # Use lexists() because exists() returns False for broken symlinks
        while broken_link.exists(follow_symlinks=False):
          time.sleep(0.01)
    finally:
      self.join_thread()

  def test_delete_mixed_files_and_directories(self):
    """Test deletion with mix of files, directories, and symlinks"""
    log_root = Path(deleter.Paths.log_root())

    # Create various items
    items_to_delete = []

    # Regular file
    regular_file = log_root / "random_file.dat"
    regular_file.write_bytes(b"data" * 1024)
    items_to_delete.append(regular_file)

    # Regular directory with openpilot segment
    seg_dir = self.make_file_with_data(self.seg_dir, self.f_type, 0.1)
    items_to_delete.append(seg_dir.parent)

    # Non-openpilot directory
    non_op_dir = log_root / "user_data"
    non_op_dir.mkdir(exist_ok=True)
    (non_op_dir / "file.txt").write_text("user file")
    items_to_delete.append(non_op_dir)

    # Broken symlink
    broken_link = log_root / "broken"
    broken_link.symlink_to("/tmp/nonexistent")
    items_to_delete.append(broken_link)

    self.start_thread()
    try:
      with Timeout(5, "Timeout waiting for all items to be deleted"):
        while any(p.exists(follow_symlinks=False) for p in items_to_delete):
          time.sleep(0.01)
    finally:
      self.join_thread()

  def test_delete_empty_directory(self):
    """Test that empty directories are handled"""
    log_root = Path(deleter.Paths.log_root())
    empty_dir = log_root / "empty_folder"
    empty_dir.mkdir(exist_ok=True)

    self.start_thread()
    try:
      with Timeout(3, "Timeout waiting for empty directory to be deleted"):
        while empty_dir.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()
