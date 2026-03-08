import os
import random
import time
import threading
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path

import openpilot.system.loggerd.deleter as deleter
from openpilot.common.timeout import Timeout, TimeoutException
from openpilot.system.hardware.hw import Paths
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

  @staticmethod
  def path_exists(path: Path) -> bool:
    return os.path.lexists(path)

  def make_random_log_root_dir(self, dirname: str, rng: random.Random) -> Path:
    root_dir = Path(Paths.log_root()) / dirname
    for i in range(rng.randint(1, 3)):
      depth = rng.randint(0, 2)
      nested_dir = "/".join(f"nested-{i}-{j}" for j in range(depth))
      file_dir = f"{dirname}/{nested_dir}" if nested_dir else dirname
      self.make_file_with_data(file_dir, f"file-{i}.bin", .01)
    return root_dir

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

  def test_delete_mixed_log_root_entries(self):
    rng = random.Random(0)

    removable_paths = [self.make_file_with_data("", f"stray-{i}.bin", .01) for i in range(3)]
    removable_paths += [self.make_random_log_root_dir(f"stray-dir-{i}", rng) for i in range(3)]

    target_file = self.make_file_with_data("", "stray-target.bin", .01)
    removable_paths.append(target_file)
    file_link = Path(Paths.log_root()) / "stray-file-link"
    file_link.symlink_to(target_file)
    removable_paths.append(file_link)

    target_dir = self.make_random_log_root_dir("stray-target-dir", rng)
    removable_paths.append(target_dir)
    dir_link = Path(Paths.log_root()) / "stray-dir-link"
    dir_link.symlink_to(target_dir, target_is_directory=True)
    removable_paths.append(dir_link)

    locked_path = self.make_file_with_data(self.seg_dir, self.f_type, .01, lock=True)

    self.start_thread()
    try:
      with Timeout(10, "Timeout waiting for mixed entries to be deleted"):
        while any(self.path_exists(path) for path in removable_paths):
          time.sleep(0.01)
    finally:
      self.join_thread()

    assert locked_path.exists(), "Locked segment deleted while cleaning mixed entries"

  def test_continue_after_listdir_failure(self, monkeypatch):
    bad_dir = self.make_random_log_root_dir("bad-dir", random.Random(1))
    file_path = self.make_file_with_data("", "cleanup-me.bin", .01)

    original_listdir = deleter.os.listdir

    def flaky_listdir(path):
      if path == str(bad_dir):
        raise OSError("boom")
      return original_listdir(path)

    monkeypatch.setattr(deleter.os, "listdir", flaky_listdir)

    self.start_thread()
    try:
      with Timeout(5, "Timeout waiting for deleter to continue after listdir failure"):
        while self.path_exists(file_path):
          time.sleep(0.01)
      assert self.del_thread.is_alive(), "Deleter thread died after listdir failure"
    finally:
      self.join_thread()

    assert bad_dir.exists(), "Failing directory should still exist"
