# pylint: skip-file

import os
import time
import tempfile
import threading
import urllib.parse
import pycurl
from hashlib import sha256
from io import BytesIO
from tenacity import retry, wait_random_exponential, stop_after_attempt

#Cache chunk size
K=1000
CHUNK_SIZE=1000 * K

PATH="/tmp/comma_download_cache/"

class URLFile(object):
  _tlocal = threading.local()

  def __init__(self, url, debug=False, cache=True):
    self._url = url
    self._pos = 0
    self._length = None
    self._local_file = None
    self._debug = debug
    #We download if we cannot cache 
    self._force_download = not cache
    try:
      self._curl = self._tlocal.curl
    except AttributeError:
      self._curl = self._tlocal.curl = pycurl.Curl()
    if not os.path.isdir(PATH):
      os.makedirs(PATH)
    if not os.path.isdir(PATH + "lengths/"):
      os.makedirs(PATH + "lengths/")
    

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if self._local_file is not None:
      os.remove(self._local_file.name)
      self._local_file.close()
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
    path_to_name = PATH + "lengths/" + str(sha256((self._url.split("?")[0]).encode('utf-8')).hexdigest()) + "_length"
    if os.path.exists(path_to_name):
      with open(path_to_name, "r") as file_length:
          content = file_length.read()
          self._length = int(content)
          return self._length

    c = self.get_curl()
    c.setopt(pycurl.URL, self._url)
    c.setopt(c.NOBODY, 1)
    c.perform()

    length = int(c.getinfo(c.CONTENT_LENGTH_DOWNLOAD))
    self._length = length
    with open(path_to_name, "w") as file_length:
      file_length.write(str(self._length))
    return length

  def read(self, ll=None):
    if self._force_download:
      return self.read_aux(ll=ll)

    file_begin = self._pos
    file_end = self._pos + ll if ll is not None else self.get_length()
    #We have to allign with chunks we store. Position is the begginiing of the latest chunk that starts before or at our file
    position = (file_begin // CHUNK_SIZE) * CHUNK_SIZE
    response = b""
    while True:
      self._pos = position
      chunk_number = self._pos / CHUNK_SIZE
      file_name = str(sha256((self._url.split("?")[0]).encode('utf-8')).hexdigest()) + "_" + str(chunk_number)
      full_path = PATH + str(file_name)
      data = None
      #If we don't have a file, download it
      if not os.path.exists(full_path):
        data = self.read_aux(ll=CHUNK_SIZE)
        with open(full_path, "wb") as new_cached_file:
          new_cached_file.write(data)
      else:
        with open(full_path, "rb") as cached_file:
          data = cached_file.read()
      
      response += data[max(0, file_begin - position) : min(CHUNK_SIZE, file_end - position)]

      position += CHUNK_SIZE
      if position >= file_end:
        self._pos = file_end
        return response

  @retry(wait=wait_random_exponential(multiplier=1, max=5), stop=stop_after_attempt(3), reraise=True)
  def read_aux(self, ll=None):
    if ll is None:
      trange = 'bytes=%d-%d' % (self._pos, self.get_length()-1)
    else:
      trange = 'bytes=%d-%d' % (self._pos, self._pos + ll - 1)

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
      if t2 - t1 > 0.1:
        print("get %s %r %.f slow" % (self._url, trange, t2 - t1))

    response_code = c.getinfo(pycurl.RESPONSE_CODE)
    #print("Response code was " + str(response_code))
    if response_code == 416:  # Requested Range Not Satisfiable
      raise Exception("Error, range out of bounds {} ({}): {}".format(response_code, self._url, repr(dats.getvalue())[:500]))
    if response_code != 206 and response_code != 200:
      raise Exception("Error {} ({}): {}".format(response_code, self._url, repr(dats.getvalue())[:500]))

    ret = dats.getvalue()
    self._pos += len(ret)
    #print("Size "+str(self.get_length())+ " Range " + trange + " gives length " + str(len(ret)))
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
