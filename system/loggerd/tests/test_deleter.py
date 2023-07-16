#!/usr/bin/env python3
import time
import threading
import unittest
from collections import namedtuple

from common.timeout import Timeout, TimeoutException
import system.loggerd.deleter as deleter
from system.loggerd.tests.loggerd_tests_common import UploaderTestCase

Stats = namedtuple("Stats", ['f_bavail', 'f_blocks', 'f_frsize'])


class TestDeleter(UploaderTestCase):
  def fake_statvfs(self, d):
    return self.fake_stats

  def setUp(self):
    self.f_type = "fcamera.hevc"
    super().setUp()
    self.fake_stats = Stats(f_bavail=0, f_blocks=10, f_frsize=4096)
    deleter.os.statvfs = self.fake_statvfs
    deleter.ROOT = str(self.root)

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

    with Timeout(5, "Timeout waiting for file to be deleted"):
      while f_path.exists():
        time.sleep(0.01)
    self.join_thread()

    self.assertFalse(f_path.exists(), "File not deleted")

  def test_delete_files_in_create_order(self):
    f_path_1 = self.make_file_with_data(self.seg_dir, self.f_type)
    time.sleep(1)
    self.seg_num += 1
    self.seg_dir = self.seg_format.format(self.seg_num)
    f_path_2 = self.make_file_with_data(self.seg_dir, self.f_type)

    self.start_thread()

    with Timeout(5, "Timeout waiting for file to be deleted"):
      while f_path_1.exists() and f_path_2.exists():
        time.sleep(0.01)

    self.join_thread()

    self.assertFalse(f_path_1.exists(), "Older file not deleted")
    self.assertTrue(f_path_2.exists(), "Newer file deleted before older file")

  def test_no_delete_when_available_space(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type)

    block_size = 4096
    available = (10 * 1024 * 1024 * 1024) / block_size  # 10GB free
    self.fake_stats = Stats(f_bavail=available, f_blocks=10, f_frsize=block_size)

    self.start_thread()

    try:
      with Timeout(2, "Timeout waiting for file to be deleted"):
        while f_path.exists():
          time.sleep(0.01)
    except TimeoutException:
      pass
    finally:
      self.join_thread()

    self.assertTrue(f_path.exists(), "File deleted with available space")

  def test_no_delete_with_lock_file(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type, lock=True)

    self.start_thread()

    try:
      with Timeout(2, "Timeout waiting for file to be deleted"):
        while f_path.exists():
          time.sleep(0.01)
    except TimeoutException:
      pass
    finally:
      self.join_thread()

    self.assertTrue(f_path.exists(), "File deleted when locked")


if __name__ == "__main__":
  unittest.main()
