#!/usr/bin/env python3
import os
import time
import threading
import unittest
import logging
import json

from selfdrive.swaglog import cloudlog
import selfdrive.loggerd.uploader as uploader

from common.xattr import getxattr

from selfdrive.loggerd.tests.loggerd_tests_common import UploaderTestCase


class TestLogHandler(logging.Handler):
  def __init__(self):
    logging.Handler.__init__(self)
    self.reset()

  def reset(self):
    self.upload_order = list()
    self.upload_ignored = list()

  def emit(self, record):
    try:
      j = json.loads(record.getMessage())
      if j["event"] == "upload_success":
        self.upload_order.append(j["key"])
      if j["event"] == "upload_ignored":
        self.upload_ignored.append(j["key"])
    except Exception:
      pass

log_handler = TestLogHandler()
cloudlog.addHandler(log_handler)


class TestUploader(UploaderTestCase):
  def setUp(self):
    super().setUp()
    log_handler.reset()

  def start_thread(self):
    self.end_event = threading.Event()
    self.up_thread = threading.Thread(target=uploader.uploader_fn, args=[self.end_event])
    self.up_thread.daemon = True
    self.up_thread.start()

  def join_thread(self):
    self.end_event.set()
    self.up_thread.join()

  def gen_files(self, lock=False, boot=True):
    f_paths = list()
    for t in ["qlog.bz2", "rlog.bz2", "dcamera.hevc", "fcamera.hevc"]:
      f_paths.append(self.make_file_with_data(self.seg_dir, t, 1, lock=lock))

    if boot:
      f_paths.append(self.make_file_with_data("boot", f"{self.seg_dir}.bz2", 1, lock=lock))
    return f_paths

  def gen_order(self, seg1, seg2, boot=True):
    keys = []
    if boot:
      keys += [f"boot/{self.seg_format.format(i)}.bz2" for i in seg1]
      keys += [f"boot/{self.seg_format2.format(i)}.bz2" for i in seg2]
    keys += [f"{self.seg_format.format(i)}/qlog.bz2" for i in seg1]
    keys += [f"{self.seg_format2.format(i)}/qlog.bz2" for i in seg2]
    return keys

  def test_upload(self):
    self.gen_files(lock=False)

    self.start_thread()
    # allow enough time that files could upload twice if there is a bug in the logic
    time.sleep(5)
    self.join_thread()

    exp_order = self.gen_order([self.seg_num], [])

    self.assertTrue(len(log_handler.upload_ignored) == 0, "Some files were ignored")
    self.assertFalse(len(log_handler.upload_order) < len(exp_order), "Some files failed to upload")
    self.assertFalse(len(log_handler.upload_order) > len(exp_order), "Some files were uploaded twice")
    for f_path in exp_order:
      self.assertTrue(getxattr(os.path.join(self.root, f_path), uploader.UPLOAD_ATTR_NAME), "All files not uploaded")

    self.assertTrue(log_handler.upload_order == exp_order, "Files uploaded in wrong order")

  def test_upload_ignored(self):
    self.set_ignore()
    self.gen_files(lock=False)

    self.start_thread()
    # allow enough time that files could upload twice if there is a bug in the logic
    time.sleep(5)
    self.join_thread()

    exp_order = self.gen_order([self.seg_num], [])

    self.assertTrue(len(log_handler.upload_order) == 0, "Some files were not ignored")
    self.assertFalse(len(log_handler.upload_ignored) < len(exp_order), "Some files failed to ignore")
    self.assertFalse(len(log_handler.upload_ignored) > len(exp_order), "Some files were ignored twice")
    for f_path in exp_order:
      self.assertTrue(getxattr(os.path.join(self.root, f_path), uploader.UPLOAD_ATTR_NAME), "All files not ignored")

    self.assertTrue(log_handler.upload_ignored == exp_order, "Files ignored in wrong order")

  def test_upload_files_in_create_order(self):
    seg1_nums = [0, 1, 2, 10, 20]
    for i in seg1_nums:
      self.seg_dir = self.seg_format.format(i)
      self.gen_files(boot=False)
    seg2_nums = [5, 50, 51]
    for i in seg2_nums:
      self.seg_dir = self.seg_format2.format(i)
      self.gen_files(boot=False)

    exp_order = self.gen_order(seg1_nums, seg2_nums, boot=False)

    self.start_thread()
    # allow enough time that files could upload twice if there is a bug in the logic
    time.sleep(5)
    self.join_thread()

    self.assertTrue(len(log_handler.upload_ignored) == 0, "Some files were ignored")
    self.assertFalse(len(log_handler.upload_order) < len(exp_order), "Some files failed to upload")
    self.assertFalse(len(log_handler.upload_order) > len(exp_order), "Some files were uploaded twice")
    for f_path in exp_order:
      self.assertTrue(getxattr(os.path.join(self.root, f_path), uploader.UPLOAD_ATTR_NAME), "All files not uploaded")

    self.assertTrue(log_handler.upload_order == exp_order, "Files uploaded in wrong order")

  def test_no_upload_with_lock_file(self):
    f_paths = self.gen_files(lock=True, boot=False)

    self.start_thread()
    # allow enough time that files should have been uploaded if they would be uploaded
    time.sleep(5)
    self.join_thread()

    for f_path in f_paths:
      self.assertFalse(getxattr(f_path, uploader.UPLOAD_ATTR_NAME), "File upload when locked")


if __name__ == "__main__":
  unittest.main(failfast=True)
