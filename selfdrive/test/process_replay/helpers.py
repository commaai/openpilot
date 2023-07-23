import os
import shutil
import uuid

from typing import List, Optional

from common.params import Params

class OpenpilotPrefix(object):
  def __init__(self, prefix: Optional[str] = None, clean_dirs_on_exit: bool = True):
    self.prefix = prefix if prefix else str(uuid.uuid4())
    self.msgq_path = os.path.join('/dev/shm', self.prefix)
    self.clean_dirs_on_exit = clean_dirs_on_exit

  def __enter__(self):
    os.environ['OPENPILOT_PREFIX'] = self.prefix
    try:
      os.mkdir(self.msgq_path)
    except FileExistsError:
      pass

  def __exit__(self, exc_type, exc_obj, exc_tb):
    if self.clean_dirs_on_exit:
      self.clean_dirs()
    del os.environ['OPENPILOT_PREFIX']
    return False

  def clean_dirs(self):
    symlink_path = Params().get_param_path()
    if os.path.exists(symlink_path):
      shutil.rmtree(os.path.realpath(symlink_path), ignore_errors=True)
      os.remove(symlink_path)
    shutil.rmtree(self.msgq_path, ignore_errors=True)


class DummySocket:
  def __init__(self):
    self.data: List[bytes] = []

  def receive(self, non_blocking: bool = False) -> Optional[bytes]:
    if non_blocking:
      return None

    return self.data.pop()

  def send(self, data: bytes):
    self.data.append(data)
