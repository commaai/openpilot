import os
from tools.lib.file_helpers import mkdirs_exists_ok

DEFAULT_CACHE_DIR = os.path.expanduser("~/.commacache")

def cache_path_for_file_path(fn, cache_prefix=None):
  dir_ = os.path.join(DEFAULT_CACHE_DIR, "local")
  mkdirs_exists_ok(dir_)
  return os.path.join(dir_, os.path.abspath(fn).replace("/", "_"))
