import os
import random
from pathlib import Path


import openpilot.system.loggerd.deleter as deleter
import openpilot.system.loggerd.uploader as uploader
from openpilot.common.params import Params
from openpilot.system.hardware.hw import Paths
from openpilot.system.loggerd.xattr_cache import setxattr


def create_random_file(file_path: Path, size_mb: float, lock: bool = False, upload_xattr: bytes = None) -> None:
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

class MockResponse:
  def __init__(self, text, status_code):
    self.text = text
    self.status_code = status_code

class MockApi:
  def __init__(self, dongle_id):
    pass

  def get(self, *args, **kwargs):
    return MockResponse('{"url": "http://localhost/does/not/exist", "headers": {}}', 200)

  def get_token(self):
    return "fake-token"

class MockApiIgnore:
  def __init__(self, dongle_id):
    pass

  def get(self, *args, **kwargs):
    return MockResponse('', 412)

  def get_token(self):
    return "fake-token"

class UploaderTestCase:
  f_type = "UNKNOWN"

  root: Path
  seg_num: int
  seg_format: str
  seg_format2: str
  seg_dir: str

  def set_ignore(self):
    uploader.Api = MockApiIgnore

  def setup_method(self):
    uploader.Api = MockApi
    uploader.fake_upload = True
    uploader.force_wifi = True
    uploader.allow_sleep = False
    self.seg_num = random.randint(1, 300)
    self.seg_format = "00000004--0ac3964c96--{}"
    self.seg_format2 = "00000005--4c4e99b08b--{}"
    self.seg_dir = self.seg_format.format(self.seg_num)

    self.params = Params()
    self.params.put("IsOffroad", "1")
    self.params.put("DongleId", "0000000000000000")

  def make_file_with_data(self, f_dir: str, fn: str, size_mb: float = .1, lock: bool = False,
                          upload_xattr: bytes = None, preserve_xattr: bytes = None) -> Path:
    file_path = Path(Paths.log_root()) / f_dir / fn
    create_random_file(file_path, size_mb, lock, upload_xattr)

    if preserve_xattr is not None:
      setxattr(str(file_path.parent), deleter.PRESERVE_ATTR_NAME, preserve_xattr)

    return file_path
