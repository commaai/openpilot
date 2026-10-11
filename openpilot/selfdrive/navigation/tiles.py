"""Authenticated range reads of the hosted PMTiles basemap."""
import hashlib
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from openpilot.common.api import Api
from openpilot.common.hardware import PC
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params

CACHE_LIMIT = 64 * 1024 * 1024


class RemoteTiles:
  def __init__(self):
    self.api = Api(Params().get('DongleId'))
    self.session = requests.Session()
    self.url = ''
    self.expires = 0
    self.etag = None
    self.size = 0
    self.cache = Path(Paths.comma_home() if PC else '/data') / 'navigation' / 'tiles'
    self.cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    self.downloads = 0
    try:
      self.header = self._fetch(0, 127)
    except Exception:
      self.close()
      raise

  def _fetch(self, offset: int, length: int) -> bytes:
    try:
      if time.time() >= self.expires - 60:  # noqa: TID251 - SAS expiry is a Unix timestamp
        response = self.api.get(f'v1/{self.api.dongle_id}/nav/tiles', access_token=self.api.get_token(), timeout=10)
        response.raise_for_status()
        metadata = response.json()
        self.url, self.expires = metadata['url'], metadata['expires']
      headers = {'Range': f'bytes={offset}-{offset + length - 1}', 'Accept-Encoding': 'identity'}
      if self.etag is not None:
        headers['If-Match'] = self.etag
      with self.session.get(self.url, headers=headers, timeout=10, stream=True) as response:
        if response.status_code != 206:
          raise OSError(f'Map tile range request failed ({response.status_code})')
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if match is None or tuple(map(int, match.groups()[:2])) != (offset, offset + length - 1):
          raise OSError('Invalid map tile range response')
        size = int(match[3])
        etag = response.headers.get('ETag')
        if not etag or (self.etag is not None and (etag != self.etag or size != self.size)):
          raise OSError('Map archive changed during read')
        data = bytearray()
        for chunk in response.iter_content(chunk_size=min(length + 1, 65536)):
          data.extend(chunk)
          if len(data) > length:
            raise OSError('Oversized map tile range response')
        if len(data) != length:
          raise OSError('Truncated map tile range response')
        self.size, self.etag = size, etag
        return bytes(data)
    except (requests.RequestException, ValueError, KeyError, TypeError, RuntimeError):
      # requests exceptions contain the signed URL; keep it out of logs.
      raise OSError('Unable to fetch map tiles') from None

  def read(self, offset: int, length: int) -> bytes:
    if offset == 0 and length == 127:
      return self.header
    if length == 0:
      return b''
    source = urlsplit(self.url)._replace(query='', fragment='').geturl()
    version = hashlib.sha256((source + self.etag).encode()).hexdigest()[:16]
    path = self.cache / f'{version}-{offset}-{length}'
    try:
      data = path.read_bytes()
      if len(data) == length:
        return data
    except FileNotFoundError:
      pass
    data = self._fetch(offset, length)
    with tempfile.NamedTemporaryFile(dir=self.cache, delete=False) as file:
      temporary = Path(file.name)
      try:
        file.write(data)
        file.flush()
        os.replace(temporary, path)
      finally:
        temporary.unlink(missing_ok=True)
    self.downloads += 1
    if self.downloads % 32 == 1:
      files = []
      for cached in self.cache.iterdir():
        if re.fullmatch(r'[0-9a-f]{16}-\d+-\d+', cached.name) is None:
          continue
        try:
          stat = cached.stat()
          files.append((stat.st_mtime, stat.st_size, cached))
        except FileNotFoundError:
          pass
      size = sum(size for _, size, _ in files)
      for _, cached_size, cached in sorted(files):
        if size <= CACHE_LIMIT:
          break
        cached.unlink(missing_ok=True)
        size -= cached_size
    return data

  def close(self):
    self.session.close()
