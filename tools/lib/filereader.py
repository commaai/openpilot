import os
from tools.lib.url_file import URLFile

DATA_PREFIX = os.getenv("DATA_PREFIX", "http://data-raw.internal/")

def FileReader(fn, debug=False):
  if fn.startswith("cd:/"):
    fn = fn.replace("cd:/", DATA_PREFIX)
  if fn.startswith("http://") or fn.startswith("https://"):
    return URLFile(fn, debug=debug)
  return open(fn, "rb")
