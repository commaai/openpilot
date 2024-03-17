#!/usr/bin/env python3
import time
import threading
import unittest
import random
import string
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

  def setUp(self):
    self.f_type = "fcamera.hevc"
    super().setUp()
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

  def get_delete_order(self, f_paths: Sequence[Path], timeout: int = 5):
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
    return deleted_order

  def assertDeleteOrder(self, f_paths: Sequence[Path], timeout: int = 5) -> None:
    self.assertEqual(self.get_delete_order(f_paths, timeout), f_paths, "Files not deleted in expected order")

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

    self.assertTrue(f_path.exists(), "File deleted with available space")

  def test_no_delete_with_lock_file(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type, lock=True)

    self.start_thread()
    start_time = time.monotonic()
    while f_path.exists() and time.monotonic() - start_time < 2:
      time.sleep(0.01)
    self.join_thread()

    self.assertTrue(f_path.exists(), "File deleted when locked")

  def generate_random_text(self, num):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=num))

  def create_random_files(self, directory: Path, num_files: int) -> list[Path]:
    file_paths = []
    for _ in range(num_files):
      file_path = directory / self.generate_random_text(10)
      file_path.write_text(self.generate_random_text(100))
      file_paths.append(file_path)
    return file_paths

  def create_directories(self, base_dir: Path, depth: int, dirs_per_level: int, files_per_dir: int) -> list[Path]:
    if depth <= 0:
      return []
    top_level_paths = []
    for _ in range(dirs_per_level):
      new_dir = base_dir / self.generate_random_text(5)
      new_dir.mkdir(parents=True, exist_ok=True)

      if depth == 1:
        top_level_paths.append(new_dir)

      self.create_random_files(new_dir, files_per_dir)
      self.create_directories(new_dir, depth - 1, dirs_per_level, files_per_dir)
    return top_level_paths

  def test_delete_files_and_dirs(self):
    created = [
      self.create_random_files(Path(Paths.log_root()), 10),
      self.create_directories(Path(Paths.log_root()), 3, 3, 3),
      [
        self.make_file_with_data(f_dir=self.seg_format.format(0), fn=self.f_type),
        self.make_file_with_data(f_dir=self.seg_format.format(1), fn=self.f_type),
        self.make_file_with_data(f_dir=self.seg_format2.format(0), fn=self.f_type),
      ],
    ]
    flattened = [item for group in created for item in group]
    delete_order = self.get_delete_order(flattened)

    index = 0
    for candidate_for_deletion in created:
      deleted = delete_order[index:index + len(candidate_for_deletion)]  # Retrieve the file that was deleted from the index of the file to be deleted.
      self.assertCountEqual(candidate_for_deletion, deleted, "Some files or directories were not deleted or were deleted in a different order.")
      index += len(candidate_for_deletion)


if __name__ == "__main__":
  unittest.main()
