import os
import posixpath
import socket
from functools import cache
from openpilot.common.retry import retry
from urllib.parse import urlparse

from openpilot.tools.lib.url_file import URLFile

DATA_ENDPOINT = os.getenv("DATA_ENDPOINT", "http://data-raw.comma.internal/")


@cache
@retry(delay=0.0)
def internal_source_available(url: str) -> bool:
  if os.path.isdir(url):
    return True

  try:
    hostname = urlparse(url).hostname
    port = urlparse(url).port or 80
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      s.settimeout(0.5)
      s.connect((hostname, port))
    return True
  except (socket.gaierror, ConnectionRefusedError):
    pass
  return False


def resolve_name(fn):
  if fn.startswith("cd:/"):
    return posixpath.join(DATA_ENDPOINT, fn[4:])
  return fn


@cache
def file_exists(fn):
  fn = resolve_name(fn)
  if fn.startswith(("http://", "https://")):
    return URLFile(fn).get_length_online() != -1
  return os.path.exists(fn)


def FileReader(fn, debug=False):
  fn = resolve_name(fn)
  if fn.startswith(("http://", "https://")):
    return URLFile(fn, debug=debug)
  return open(fn, "rb")
