import os
import urllib.parse
from openpilot.common.file_helpers import mkdirs_exists_ok

DEFAULT_CACHE_DIR = os.getenv("CACHE_ROOT", os.path.expanduser("~/.commacache"))

def cache_path_for_file_path(fn, cache_dir=DEFAULT_CACHE_DIR):
  dir_ = os.path.join(cache_dir, "local")
  mkdirs_exists_ok(dir_)
  fn_parsed = urllib.parse.urlparse(fn)
  if fn_parsed.scheme == '':
    cache_fn = os.path.abspath(fn).replace("/", "_")
  else:
    cache_fn = f'{fn_parsed.hostname}_{fn_parsed.path.replace("/", "_")}'
  return os.path.join(dir_, cache_fn)
