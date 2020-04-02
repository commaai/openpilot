import os
import sys
import time
import tempfile
import threading
import urllib.parse
import pycurl
import hashlib
from io import BytesIO
from tenacity import retry, wait_random_exponential, stop_after_attempt
from common.file_helpers import mkdirs_exists_ok, atomic_write_in_dir

class URLFile(object):
  _tlocal = threading.local()

  def __init__(self, url, debug=False):
    self._url = url
    self._pos = 0
    self._local_file = None
    self._debug = debug

    try:
      self._curl = self._tlocal.curl
    except AttributeError:
      self._curl = self._tlocal.curl = pycurl.Curl()

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    if self._local_file is not None:
      os.remove(self._local_file.name)
      self._local_file.close()
      self._local_file = None

  @retry(wait=wait_random_exponential(multiplier=1, max=5), stop=stop_after_attempt(3), reraise=True)
  def read(self, ll=None):
    if ll is None:
      trange = 'bytes=%d-' % self._pos
    else:
      trange = 'bytes=%d-%d' % (self._pos, self._pos+ll-1)

    dats = BytesIO()
    c = self._curl
    c.setopt(pycurl.URL, self._url)
    c.setopt(pycurl.WRITEDATA, dats)
    c.setopt(pycurl.NOSIGNAL, 1)
    c.setopt(pycurl.TIMEOUT_MS, 500000)
    c.setopt(pycurl.HTTPHEADER, ["Range: " + trange, "Connection: keep-alive"])
    c.setopt(pycurl.FOLLOWLOCATION, True)

    if self._debug:
      print("downloading", self._url)
      def header(x):
        if b'MISS' in x:
          print(x.strip())
      c.setopt(pycurl.HEADERFUNCTION, header)
      def test(debug_type, debug_msg):
       print("  debug(%d): %s" % (debug_type, debug_msg.strip()))
      c.setopt(pycurl.VERBOSE, 1)
      c.setopt(pycurl.DEBUGFUNCTION, test)
      t1 = time.time()

    c.perform()

    if self._debug:
      t2 = time.time()
      if t2-t1 > 0.1:
        print("get %s %r %.f slow" % (self._url, trange, t2-t1))

    response_code = c.getinfo(pycurl.RESPONSE_CODE)
    if response_code == 416: #  Requested Range Not Satisfiable
      return ""
    if response_code != 206 and response_code != 200:
      raise Exception("Error {} ({}): {}".format(response_code, self._url, repr(dats.getvalue())[:500]))


    ret = dats.getvalue()
    self._pos += len(ret)
    return ret

  def seek(self, pos):
    self._pos = pos

  @property
  def name(self):
    """Returns a local path to file with the URLFile's contents.

       This can be used to interface with modules that require local files.
    """
    if self._local_file is None:
      _, ext = os.path.splitext(urllib.parse.urlparse(self._url).path)
      local_fd, local_path = tempfile.mkstemp(suffix=ext)
      try:
        os.write(local_fd, self.read())
        local_file = open(local_path, "rb")
      except:
        os.remove(local_path)
        raise
      finally:
        os.close(local_fd)

      self._local_file = local_file
      self.read = self._local_file.read
      self.seek = self._local_file.seek

    return self._local_file.name
