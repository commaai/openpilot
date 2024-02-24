import os
import socket
from urllib.parse import urlparse

from openpilot.tools.lib.url_file import URLFile

DATA_ENDPOINT = os.getenv("DATA_ENDPOINT", "http://data-raw.comma.internal/")


def internal_source_available():
  try:
    hostname = urlparse(DATA_ENDPOINT).hostname
    if hostname:
      socket.gethostbyname(hostname)
      return True
  except socket.gaierror:
    pass
  return False


def resolve_name(fn):
  if fn.startswith("cd:/"):
    return fn.replace("cd:/", DATA_ENDPOINT)
  return fn


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
