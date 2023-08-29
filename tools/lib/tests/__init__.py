import tempfile

from unittest import mock


def temporary_mock_dir(mock_path):
  def decorator(func):
    def wrapper(*args, **kwargs):
      with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir_patch = mock.patch(mock_path, temp_dir)
        cache_dir_patch.start()
        func(*args, **kwargs)
        cache_dir_patch.stop()
      return wrapper
  return decorator

temporary_cache_dir = temporary_mock_dir("openpilot.tools.lib.url_file.CACHE_DIR")