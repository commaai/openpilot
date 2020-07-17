# pylint: skip-file

import os
import tempfile
import urllib.parse
import pycurl
from io import BytesIO
from tenacity import retry, wait_random_exponential, stop_after_attempt

from multiprocessing import Pool


class URLFile(object):
  def __init__(self, url, debug=False):
    self._url = url
    self._pos = 0
    self._local_file = None

  def get_curl(self):
    curl = pycurl.Curl()
    curl.setopt(pycurl.NOSIGNAL, 1)
    curl.setopt(pycurl.TIMEOUT_MS, 500000)
    curl.setopt(pycurl.FOLLOWLOCATION, True)
    return curl

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if self._local_file is not None:
      os.remove(self._local_file.name)
      self._local_file.close()
      self._local_file = None

  @retry(wait=wait_random_exponential(multiplier=1, max=5), stop=stop_after_attempt(3), reraise=True)
  def get_length(self):
    c = self.get_curl()
    c.setopt(pycurl.URL, self._url)
    c.setopt(c.NOBODY, 1)
    c.perform()

    return int(c.getinfo(c.CONTENT_LENGTH_DOWNLOAD))

  @retry(wait=wait_random_exponential(multiplier=1, max=5), stop=stop_after_attempt(3), reraise=True)
  def download_chunk(self, start, end=None):
    if end is None:
      trange = f'bytes={start}-'
    else:
      trange = f'bytes={start}-{end}'

    dats = BytesIO()
    c = self.get_curl()
    c.setopt(pycurl.URL, self._url)
    c.setopt(pycurl.WRITEDATA, dats)
    c.setopt(pycurl.HTTPHEADER, ["Range: " + trange, "Connection: keep-alive"])
    c.perform()

    response_code = c.getinfo(pycurl.RESPONSE_CODE)
    if response_code == 416:  # Requested Range Not Satisfiable
      return b""
    if response_code != 206 and response_code != 200:
      raise Exception("Error {} ({}): {}".format(response_code, self._url, repr(dats.getvalue())[:500]))

    return dats.getvalue()

  def read(self, ll=None):
    start = self._pos
    end = None if ll is None else self._pos + ll - 1
    threads = int(os.environ.get("COMMA_PARALLEL_DOWNLOADS", "0"))

    if threads > 0:
      # Multithreaded download
      end = self.get_length() if end is None else end
      chunk_size = (end - start) // threads
      chunks = [
        (start + chunk_size * i,
         start + chunk_size * (i + 1) if i != threads - 1 else end)
        for i in range(threads)]

      with Pool(threads) as pool:
        ret = b"".join(pool.starmap(self.download_chunk, chunks))
    else:
      # Single threaded download
      ret = self.download_chunk(start, end)

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
      except Exception:
        os.remove(local_path)
        raise
      finally:
        os.close(local_fd)

      self._local_file = local_file
      self.read = self._local_file.read
      self.seek = self._local_file.seek

    return self._local_file.name
