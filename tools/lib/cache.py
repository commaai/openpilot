import os
import urllib.parse
from openpilot.common.file_helpers import mkdirs_exists_ok

DEFAULT_CACHE_DIR = os.path.expanduser("~/.commacache")

def cache_path_for_file_path(fn, cache_prefix=None):
  dir_ = os.path.join(DEFAULT_CACHE_DIR, "local")
  mkdirs_exists_ok(dir_)
  fn_parsed = urllib.parse.urlparse(fn)
  if fn_parsed.scheme == '':
    cache_fn = os.path.abspath(fn).replace("/", "_")
  else:
    cache_fn = f'{fn_parsed.hostname}_{fn_parsed.path.replace("/", "_")}'
  return os.path.join(dir_, cache_fn)
