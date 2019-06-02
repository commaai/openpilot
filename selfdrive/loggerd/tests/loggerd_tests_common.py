import os
import errno
import shutil
import random
import tempfile
import unittest

import selfdrive.loggerd.uploader as uploader


def create_random_file(file_path, size_mb, lock=False):
    try:
      os.mkdir(os.path.dirname(file_path))
    except OSError:
      pass

    lock_path = file_path + ".lock"
    os.close(os.open(lock_path, os.O_CREAT | os.O_EXCL))

    chunks = 128
    chunk_bytes = int(size_mb * 1024 * 1024 / chunks)
    data = os.urandom(chunk_bytes)

    with open(file_path, 'wb') as f:
      for _ in range(chunks):
        f.write(data)

    if not lock:
        os.remove(lock_path)


class UploaderTestCase(unittest.TestCase):
  f_type = "UNKNOWN"

  def setUp(self):
    self.root = tempfile.mkdtemp()
    uploader.ROOT = self.root  # Monkey patch root dir
    self.seg_num = random.randint(1, 300)
    self.seg_format = "2019-04-18--12-52-54--{}"
    self.seg_dir = self.seg_format.format(self.seg_num)

  def tearDown(self):
    try:
      shutil.rmtree(self.root)
    except OSError as e:
      if e.errno != errno.ENOENT:
        raise

  def make_file_with_data(self, f_dir, fn, size_mb=.1, lock=False):
    file_path = os.path.join(self.root, f_dir, fn)
    create_random_file(file_path, size_mb)

    return file_path
