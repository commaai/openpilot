import os
import shutil
import uuid

from typing import Optional

from openpilot.common.params import Params
from openpilot.system.hardware.hw import Paths

class OpenpilotPrefix:
  def __init__(self, prefix: Optional[str] = None, clean_dirs_on_exit: bool = True):
    self.prefix = prefix if prefix else str(uuid.uuid4().hex[0:15])
    self.msgq_path = os.path.join('/dev/shm', self.prefix)
    self.clean_dirs_on_exit = clean_dirs_on_exit

  def __enter__(self):
    self.original_prefix = os.environ.get('OPENPILOT_PREFIX', None)
    os.environ['OPENPILOT_PREFIX'] = self.prefix
    try:
      os.mkdir(self.msgq_path)
    except FileExistsError:
      pass
    os.makedirs(Paths.log_root(), exist_ok=True)

    return self

  def __exit__(self, exc_type, exc_obj, exc_tb):
    if self.clean_dirs_on_exit:
      self.clean_dirs()
    try:
      del os.environ['OPENPILOT_PREFIX']
      if self.original_prefix is not None:
        os.environ['OPENPILOT_PREFIX'] = self.original_prefix
    except KeyError:
      pass
    return False

  def clean_dirs(self):
    symlink_path = Params().get_param_path()
    if os.path.exists(symlink_path):
      shutil.rmtree(os.path.realpath(symlink_path), ignore_errors=True)
      os.remove(symlink_path)
    shutil.rmtree(self.msgq_path, ignore_errors=True)
    shutil.rmtree(Paths.log_root(), ignore_errors=True)
    shutil.rmtree(Paths.download_cache_root(), ignore_errors=True)
    shutil.rmtree(Paths.comma_home(), ignore_errors=True)
