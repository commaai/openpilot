import tempfile

from unittest import mock

def temporary_cache_dir(func):
  def wrapper(*args, **kwargs):
    with tempfile.TemporaryDirectory() as temp_dir:
      cache_dir_patch = mock.patch("openpilot.tools.lib.url_file.CACHE_DIR", temp_dir)
      cache_dir_patch.start()
      func(*args, **kwargs)
      cache_dir_patch.stop()
  return wrapper