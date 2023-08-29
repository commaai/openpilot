import tempfile

from unittest import mock

def temporary_swaglog_dir(func):
  def wrapper(*args, **kwargs):
    with tempfile.TemporaryDirectory() as temp_dir:
      swaglog_dir_patch = mock.patch("openpilot.system.swaglog.SWAGLOG_DIR", temp_dir)
      swaglog_dir_patch.start()
      func(*args, temp_dir, **kwargs)
      swaglog_dir_patch.stop()
  return wrapper