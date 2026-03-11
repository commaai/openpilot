import os
import random
import time
import threading
from collections import namedtuple
from pathlib import Path
from collections.abc import Sequence

import openpilot.system.loggerd.deleter as deleter
from openpilot.common.timeout import Timeout, TimeoutException
from openpilot.system.loggerd.tests.loggerd_tests_common import UploaderTestCase, create_random_file
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

  def assertDeleted(self, paths: Sequence[Path], timeout: int = 5) -> None:
    self.start_thread()
    try:
      with Timeout(timeout, "Timeout waiting for paths to be deleted"):
        while any(os.path.lexists(path) for path in paths):
          time.sleep(0.01)
    finally:
      self.join_thread()

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

  def test_delete_random_mixed_entries(self):
    rng = random.Random(0)
    root = Path(Paths.log_root())
    paths_to_delete: list[Path] = []

    for i in range(6):
      file_path = root / f"stray_file_{i}.bin"
      create_random_file(file_path, 0.01)
      paths_to_delete.append(file_path)

    for i in range(4):
      dir_path = root / f"stray_dir_{i}"
      current = dir_path
      for depth in range(rng.randint(1, 3)):
        current /= f"nested_{depth}"
        current.mkdir(parents=True, exist_ok=True)
        for file_idx in range(rng.randint(1, 3)):
          create_random_file(current / f"nested_file_{depth}_{file_idx}.bin", 0.01)
      paths_to_delete.append(dir_path)

    external_target = root.parent / "deleter_symlink_target.bin"
    create_random_file(external_target, 0.01)
    symlink_path = root / "stray_symlink"
    os.symlink(external_target, symlink_path)
    paths_to_delete.append(symlink_path)

    try:
      self.assertDeleted(paths_to_delete, timeout=10)
      assert external_target.exists(), "Deleting a symlink should not remove its target"
    finally:
      if external_target.exists():
        external_target.unlink()

  def test_continue_after_entry_error(self, monkeypatch):
    root = Path(Paths.log_root())
    bad_dir = root / "0_bad_dir"
    good_dir = root / "1_good_dir"

    create_random_file(bad_dir / "nested" / "bad.bin", 0.01)
    create_random_file(good_dir / "nested" / "good.bin", 0.01)

    original_listdir = deleter.os.listdir

    def fake_listdir(path):
      if path == str(bad_dir):
        raise PermissionError("permission denied")
      return original_listdir(path)

    monkeypatch.setattr(deleter.os, "listdir", fake_listdir)

    self.start_thread()
    try:
      with Timeout(5, "Timeout waiting for good_dir to be deleted"):
        while good_dir.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()

    assert bad_dir.exists(), "The failing directory should remain after a deletion error"
