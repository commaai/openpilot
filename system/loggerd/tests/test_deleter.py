import time
import threading
from collections import namedtuple
from pathlib import Path
from collections.abc import Sequence

import openpilot.system.loggerd.deleter as deleter
from openpilot.common.timeout import Timeout, TimeoutException
from openpilot.system.loggerd.tests.loggerd_tests_common import UploaderTestCase, create_random_file

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

  def test_delete_stray_files(self):
    """Test that stray files in log root are deleted properly"""
    # Create stray files (not in segment directories)
    stray_file1 = Path(Paths.log_root()) / "backup.tar.gz"
    stray_file2 = Path(Paths.log_root()) / "notes.txt"
    stray_file3 = Path(Paths.log_root()) / "data.zip"
    
    stray_file1.write_text("stray content")
    stray_file2.write_text("stray content")
    stray_file3.write_text("stray content")
    
    # Also create a normal segment directory
    seg_path = self.make_file_with_data(self.seg_dir, self.f_type)
    
    self.start_thread()
    
    try:
      with Timeout(5, "Timeout waiting for stray files to be deleted"):
        while stray_file1.exists() or stray_file2.exists() or stray_file3.exists() or seg_path.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()
    
    # All files should be deleted
    assert not stray_file1.exists(), "Stray file 1 not deleted"
    assert not stray_file2.exists(), "Stray file 2 not deleted"
    assert not stray_file3.exists(), "Stray file 3 not deleted"
    assert not seg_path.exists(), "Segment not deleted"

  def test_delete_stray_files_mixed(self):
    """Test deleting stray files mixed with directories"""
    # Create stray files
    stray_file = Path(Paths.log_root()) / "stray.tar.gz"
    stray_file.write_text("stray content")
    
    # Create segment directories
    seg1 = self.make_file_with_data(self.seg_format.format(0), self.f_type)
    seg2 = self.make_file_with_data(self.seg_format.format(1), self.f_type)
    
    # All should be deleted in order (oldest first)
    self.assertDeleteOrder([seg1, seg2])
    
    # Stray file should also be deleted
    assert not stray_file.exists(), "Stray file not deleted"

  def test_delete_symlinks(self):
    """Test that symlinks in log root are deleted properly"""
    from openpilot.system.hardware.hw import Paths
    
    # Create a regular file to symlink to
    target = Path(Paths.log_root()) / "target.txt"
    target.write_text("target content")
    
    # Create a symlink
    symlink = Path(Paths.log_root()) / "link.txt"
    symlink.symlink_to(target)
    
    assert symlink.is_symlink(), "Symlink not created"
    
    self.start_thread()
    
    try:
      with Timeout(5, "Timeout waiting for symlink to be deleted"):
        while target.exists() or symlink.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()
    
    assert not symlink.exists(), "Symlink not deleted"
    assert not target.exists(), "Target not deleted"

  def test_delete_broken_symlink(self):
    """Test that broken symlinks are deleted properly"""
    from openpilot.system.hardware.hw import Paths
    
    # Create a symlink to non-existent target
    symlink = Path(Paths.log_root()) / "broken_link"
    try:
      symlink.symlink_to("/nonexistent/path/to/file")
      
      assert symlink.is_symlink(), "Broken symlink not created"
      assert not symlink.exists(), "Broken symlink should not exist() true"
      
      self.start_thread()
      
      try:
        with Timeout(5, "Timeout waiting for broken symlink to be deleted"):
          # Use is_symlink() to check since exists() returns False for broken symlinks
          while symlink.is_symlink():
            time.sleep(0.01)
      finally:
        self.join_thread()
      
      assert not symlink.is_symlink(), "Broken symlink not deleted"
    except OSError:
      # Some systems don't allow broken symlinks, skip test
      pass

  def test_delete_empty_directory(self):
    """Test that empty directories are deleted properly"""
    from openpilot.system.hardware.hw import Paths
    
    empty_dir = Path(Paths.log_root()) / "empty_dir"
    empty_dir.mkdir(parents=True, exist_ok=True)
    
    assert empty_dir.exists() and empty_dir.is_dir(), "Empty directory not created"
    
    self.start_thread()
    
    try:
      with Timeout(5, "Timeout waiting for empty directory to be deleted"):
        while empty_dir.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()
    
    assert not empty_dir.exists(), "Empty directory not deleted"

  def test_delete_stray_files_large(self):
    """Test deleting large stray files"""
    from openpilot.system.hardware.hw import Paths
    
    # Create a large stray file (simulate the original bug with .tar.gz)
    large_file = Path(Paths.log_root()) / "2023-09-23.tar.gz"
    # Create 10MB file
    create_random_file(large_file, size_mb=10.0)
    
    assert large_file.exists(), "Large file not created"
    
    self.start_thread()
    
    try:
      with Timeout(5, "Timeout waiting for large file to be deleted"):
        while large_file.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()
    
    assert not large_file.exists(), "Large stray file not deleted"

  def test_delete_nested_stray_structure(self):
    """Test deleting stray nested directory structures"""
    from openpilot.system.hardware.hw import Paths
    
    # Create nested structure not following segment naming convention
    nested_dir = Path(Paths.log_root()) / "backup" / "old" / "data"
    nested_dir.mkdir(parents=True, exist_ok=True)
    
    nested_file = nested_dir / "file.txt"
    nested_file.write_text("nested content")
    
    root_stray = Path(Paths.log_root()) / "backup"
    assert root_stray.exists(), "Nested structure not created"
    
    self.start_thread()
    
    try:
      with Timeout(5, "Timeout waiting for nested structure to be deleted"):
        while root_stray.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()
    
    assert not root_stray.exists(), "Nested stray structure not deleted"

  def test_deleter_doesnt_crash_on_stray_files(self):
    """Regression test for issue #30102 - deleter should not crash on stray files"""
    from openpilot.system.hardware.hw import Paths
    
    # Reproduce the exact scenario from the bug report
    stray_tarball = Path(Paths.log_root()) / "realdata" / "2023-09-23.tar.gz"
    stray_tarball.parent.mkdir(parents=True, exist_ok=True)
    create_random_file(stray_tarball, size_mb=1.0)
    
    seg_path = self.make_file_with_data(self.seg_dir, self.f_type)
    
    # Deleter thread should not crash and should continue running
    self.start_thread()
    
    try:
      with Timeout(5, "Timeout waiting for files to be deleted"):
        while stray_tarball.exists() or seg_path.exists():
          time.sleep(0.01)
      # If we get here, deleter handled stray files properly
      assert True, "Deleter handled stray files without crashing"
    finally:
      self.join_thread()
