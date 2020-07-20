# pylint: skip-file

import os
import pycurl
from tools.lib.url_file import URLFile
from io import BytesIO
from tenacity import retry, wait_random_exponential, stop_after_attempt

from multiprocessing import Pool


class URLFileParallel(URLFile):
  def __init__(self, url, debug=False):
    self._length = None
    self._url = url
    self._pos = 0
    self._local_file = None

  def get_curl(self):
    curl = pycurl.Curl()
    curl.setopt(pycurl.NOSIGNAL, 1)
    curl.setopt(pycurl.TIMEOUT_MS, 500000)
    curl.setopt(pycurl.FOLLOWLOCATION, True)
    return curl

  @retry(wait=wait_random_exponential(multiplier=1, max=5), stop=stop_after_attempt(3), reraise=True)
  def get_length(self):
    if self._length is not None:
      return self._length

    c = self.get_curl()
    c.setopt(pycurl.URL, self._url)
    c.setopt(c.NOBODY, 1)
    c.perform()

    length = int(c.getinfo(c.CONTENT_LENGTH_DOWNLOAD))
    self._length = length
    return length

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
    if response_code != 206 and response_code != 200:
      raise Exception("Error {} ({}): {}".format(response_code, self._url, repr(dats.getvalue())[:500]))

    return dats.getvalue()

  def read(self, ll=None):
    start = self._pos
    end = None if ll is None else self._pos + ll - 1
    max_threads = int(os.environ.get("COMMA_PARALLEL_DOWNLOADS", "0"))

    end = self.get_length() if end is None else end
    threads = min((end - start) // (512 * 1024), max_threads)  # At least 512k per thread

    if threads > 1:
      chunk_size = (end - start) // threads
      chunks = [
        (start + chunk_size * i,
         start + chunk_size * (i + 1) - 1 if i != threads - 1 else end)
        for i in range(threads)]

      with Pool(threads) as pool:
        ret = b"".join(pool.starmap(self.download_chunk, chunks))
    else:
      ret = self.download_chunk(start, end)

    self._pos += len(ret)
    return ret
