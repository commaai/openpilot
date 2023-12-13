import os
import random
import unittest
from pathlib import Path
from typing import Optional
from openpilot.system.hardware.hw import Paths

import openpilot.system.loggerd.deleter as deleter
import openpilot.system.loggerd.uploader as uploader
from openpilot.system.loggerd.xattr_cache import setxattr


def create_random_file(file_path: Path, size_mb: float, lock: bool = False, upload_xattr: Optional[bytes] = None) -> None:
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

  if upload_xattr is not None:
    setxattr(str(file_path), uploader.UPLOAD_ATTR_NAME, upload_xattr)

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
  # Follow a pattern similar to selfdrive/athena/tests/helpers.py#MockParams
  default_params = {
    "DongleId": b"0000000000000000",
    "IsOffroad": b"1",
    "AllowMeteredUploads": b"1",
  }
  params = default_params.copy()

  @staticmethod
  def restore_defaults():
    MockParams.params = MockParams.default_params.copy()

  def get(self, k, block=False, encoding=None):
    val = MockParams.params.get(k)

    if encoding is not None:
      return val.decode(encoding)
    else:
      return val

  def get_bool(self, k):
    val = MockParams.params.get(k)
    return (val == b'1')

  def put(self, k, v):
    if k not in MockParams.params:
      raise KeyError(f"key: {k} not in MockParams")
    MockParams.params[k] = v

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
    uploader.Api = MockApi
    uploader.Params = MockParams
    uploader.fake_upload = True
    uploader.force_wifi = True
    uploader.allow_sleep = False

    # Follow a pattern similar to selfdrive/athena/tests/helpers.py#TestAthenadMethods
    MockParams.restore_defaults()

    self.seg_num = random.randint(1, 300)
    self.seg_format = "2019-04-18--12-52-54--{}"
    self.seg_format2 = "2019-05-18--11-22-33--{}"
    self.seg_dir = self.seg_format.format(self.seg_num)

  def make_file_with_data(self, f_dir: str, fn: str, size_mb: float = .1, lock: bool = False,
                          upload_xattr: Optional[bytes] = None, preserve_xattr: Optional[bytes] = None) -> Path:
    file_path = Path(Paths.log_root()) / f_dir / fn
    create_random_file(file_path, size_mb, lock, upload_xattr)

    if preserve_xattr is not None:
      setxattr(str(file_path.parent), deleter.PRESERVE_ATTR_NAME, preserve_xattr)

    return file_path
