import os
import errno
import shutil
import random
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import system.loggerd.uploader as uploader


def create_random_file(file_path: Path, size_mb: float, lock: bool = False, xattr: Optional[bytes] = None) -> None:
  file_path.parent.mkdir(parents=True, exist_ok=True)

  if lock:
    lock_path = str(file_path) + ".lock"
    os.close(os.open(lock_path, os.O_CREAT | os.O_EXCL))

  chunks = 128
  chunk_bytes = int(size_mb * 1024 * 1024 / chunks)
  data = os.urandom(chunk_bytes)

  with open(file_path, "wb") as f:
    for _ in range(chunks):
      f.write(data)

  if xattr is not None:
    uploader.setxattr(str(file_path), uploader.UPLOAD_ATTR_NAME, xattr)

class MockResponse():
  def __init__(self, text, status_code):
    self.text = text
    self.status_code = status_code

class MockApi():
  def __init__(self, dongle_id):
    pass

  def get(self, *args, **kwargs):
    return MockResponse('{"url": "http://localhost/does/not/exist", "headers": {}}', 200)

  def get_token(self):
    return "fake-token"

class MockApiIgnore():
  def __init__(self, dongle_id):
    pass

  def get(self, *args, **kwargs):
    return MockResponse('', 412)

  def get_token(self):
    return "fake-token"

class MockParams():
  def __init__(self):
    self.params = {
      "DongleId": b"0000000000000000",
      "IsOffroad": b"1",
    }

  def get(self, k, block=False, encoding=None):
    val = self.params[k]

    if encoding is not None:
      return val.decode(encoding)
    else:
      return val

  def get_bool(self, k):
    val = self.params[k]
    return (val == b'1')

class UploaderTestCase(unittest.TestCase):
  f_type = "UNKNOWN"

  root: Path
  seg_num: int
  seg_format: str
  seg_format2: str
  seg_dir: str

  def set_ignore(self):
    uploader.Api = MockApiIgnore

  def setUp(self):
    self.root = Path(tempfile.mkdtemp())
    uploader.ROOT = str(self.root)  # Monkey patch root dir
    uploader.Api = MockApi
    uploader.Params = MockParams
    uploader.fake_upload = True
    uploader.force_wifi = True
    uploader.allow_sleep = False
    self.seg_num = random.randint(1, 300)
    self.seg_format = "2019-04-18--12-52-54--{}"
    self.seg_format2 = "2019-05-18--11-22-33--{}"
    self.seg_dir = self.seg_format.format(self.seg_num)

  def tearDown(self):
    try:
      shutil.rmtree(self.root)
    except OSError as e:
      if e.errno != errno.ENOENT:
        raise

  def make_file_with_data(self, f_dir: str, fn: str, size_mb: float = .1, lock: bool = False,
                          xattr: Optional[bytes] = None) -> Path:
    file_path = self.root / f_dir / fn
    create_random_file(file_path, size_mb, lock, xattr)

    return file_path
