import os
import socket
from urllib.parse import urlparse
from openpilot.tools.lib.url_file import URLFile

DATA_ENDPOINT = os.getenv("DATA_ENDPOINT", "http://data-raw.comma.internal/")


def is_data_endpoint_accessible():
  hostname = urlparse(DATA_ENDPOINT).hostname
  if not len(hostname):
    return False

  try:
    _ = socket.gethostbyname(hostname)
    return True
  except OSError:
    return False


def resolve_name(fn):
  if fn.startswith("cd:/"):
    return fn.replace("cd:/", DATA_ENDPOINT)
  return fn


def FileReader(fn, debug=False):
  fn = resolve_name(fn)
  if fn.startswith(("http://", "https://")):
    return URLFile(fn, debug=debug)
  return open(fn, "rb")
