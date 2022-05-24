import os
import shutil
import uuid

from common.params import Params

def setup_prefix():
  os.environ['OPENPILOT_PREFIX'] = str(uuid.uuid4())
  msgq_path = os.path.join('/dev/shm', os.environ['OPENPILOT_PREFIX'])
  try:
    os.mkdir(msgq_path)
  except FileExistsError:
    pass


def teardown_prefix():
  if not os.environ.get("OPENPILOT_PREFIX", 0):
    return
  symlink_path = Params().get_param_path()
  if os.path.exists(symlink_path):
    shutil.rmtree(os.path.realpath(symlink_path), ignore_errors=True)
    os.remove(symlink_path)
  msgq_path = os.path.join('/dev/shm', os.environ['OPENPILOT_PREFIX'])
  shutil.rmtree(msgq_path, ignore_errors=True)

