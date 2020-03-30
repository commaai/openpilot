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

  def emit(self, record):
    try:
      j = json.loads(record.message)
      if j["event"] == "upload_success":
        self.upload_order.append(j["key"])
    except Exception:
      pass

log_handler = TestLogHandler()
cloudlog.addHandler(log_handler)

class TestUploader(UploaderTestCase):
  def setUp(self):
    super(TestUploader, self).setUp()
    log_handler.reset()

  def tearDown(self):
    super(TestUploader, self).tearDown()

  def start_thread(self):
    self.end_event = threading.Event()
    self.up_thread = threading.Thread(target=uploader.uploader_fn, args=[self.end_event])
    self.up_thread.daemon = True
    self.up_thread.start()

  def join_thread(self):
    self.end_event.set()
    self.up_thread.join()

  def gen_files(self, lock=False):
    f_paths = list()
    for t in ["bootlog.bz2", "qlog.bz2", "rlog.bz2", "dcamera.hevc", "fcamera.hevc"]:
      f_paths.append(self.make_file_with_data(self.seg_dir, t, 1, lock=lock))
    return f_paths

  def gen_order(self, seg1, seg2):
    keys = [f"{self.seg_format.format(i)}/qlog.bz2" for i in seg1]
    keys += [f"{self.seg_format2.format(i)}/qlog.bz2" for i in seg2]
    for i in seg1:
      keys += [f"{self.seg_format.format(i)}/{f}" for f in ['rlog.bz2','fcamera.hevc','dcamera.hevc']]
    for i in seg2:
      keys += [f"{self.seg_format2.format(i)}/{f}" for f in ['rlog.bz2','fcamera.hevc','dcamera.hevc']]
    keys += [f"{self.seg_format.format(i)}/bootlog.bz2" for i in seg1]
    keys += [f"{self.seg_format2.format(i)}/bootlog.bz2" for i in seg2]
    return keys

  def test_upload(self):
    f_paths = self.gen_files(lock=False)

    self.start_thread()
    # allow enough time that files could upload twice if there is a bug in the logic
    time.sleep(5)
    self.join_thread()

    self.assertFalse(len(log_handler.upload_order) < len(f_paths), "Some files failed to upload")
    self.assertFalse(len(log_handler.upload_order) > len(f_paths), "Some files were uploaded twice")
    for f_path in f_paths:
      self.assertTrue(getxattr(f_path, uploader.UPLOAD_ATTR_NAME), "All files not uploaded")
    exp_order = self.gen_order([self.seg_num], [])
    self.assertTrue(log_handler.upload_order == exp_order, "Files uploaded in wrong order")

  def test_upload_files_in_create_order(self):
    f_paths = list()
    seg1_nums = [0,1,2,10,20]
    for i in seg1_nums:
      self.seg_dir = self.seg_format.format(i)
      f_paths += self.gen_files()
    seg2_nums = [5,50,51]
    for i in seg2_nums:
      self.seg_dir = self.seg_format2.format(i)
      f_paths += self.gen_files()

    self.start_thread()
    # allow enough time that files could upload twice if there is a bug in the logic
    time.sleep(5)
    self.join_thread()

    self.assertFalse(len(log_handler.upload_order) < len(f_paths), "Some files failed to upload")
    self.assertFalse(len(log_handler.upload_order) > len(f_paths), "Some files were uploaded twice")
    for f_path in f_paths:
      self.assertTrue(getxattr(f_path, uploader.UPLOAD_ATTR_NAME), "All files not uploaded")
    exp_order = self.gen_order(seg1_nums, seg2_nums)
    self.assertTrue(log_handler.upload_order == exp_order, "Files uploaded in wrong order")

  def test_no_upload_with_lock_file(self):
    f_paths = self.gen_files(lock=True)

    self.start_thread()
    # allow enough time that files should have been uploaded if they would be uploaded
    time.sleep(5)
    self.join_thread()

    for f_path in f_paths:
      self.assertFalse(getxattr(f_path, uploader.UPLOAD_ATTR_NAME), "File upload when locked")


if __name__ == "__main__":
  unittest.main()
