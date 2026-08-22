#!/usr/bin/env python3
"""
CLI tool for downloading files and querying the comma API.
Called by C++ replay/cabana via subprocess.

Subcommands:
  route-files <route>    - Get route file URLs as JSON
  download <url>         - Download/decompress URL to local cache, print local path
  decompress <path>      - Decompress a local log file, print temporary path
  devices                - List user's devices as JSON
  device-routes <did>    - List routes for a device as JSON
"""
import argparse
import bz2
import hashlib
import json
import os
import shutil
import sys
import tempfile

import zstandard as zstd

from openpilot.common.hardware.hw import Paths
from openpilot.tools.lib.api import CommaApi, UnauthorizedError, APIError
from openpilot.tools.lib.auth_config import get_token
from openpilot.tools.lib.url_file import URLFile


def api_call(func):
  """Run an API call, outputting JSON result or error to stdout."""
  try:
    result = func(CommaApi(get_token()))
    json.dump(result, sys.stdout)
  except UnauthorizedError:
    json.dump({"error": "unauthorized"}, sys.stdout)
  except APIError as e:
    error = "not_found" if getattr(e, 'status_code', 0) == 404 else str(e)
    json.dump({"error": error}, sys.stdout)
  except Exception as e:
    json.dump({"error": str(e)}, sys.stdout)
  sys.stdout.write("\n")
  sys.stdout.flush()


def cache_file_path(url, compression=None):
  url_without_query = url.split("?")[0]
  if compression:
    url_without_query = f"decompressed-{compression}:{url_without_query}"
  return os.path.join(Paths.download_cache_root(), hashlib.sha256(url_without_query.encode()).hexdigest())


def compression_type(data):
  if data.startswith(b'BZh'):
    return 'bz2'
  if data.startswith(b'\x28\xb5\x2f\xfd'):
    return 'zst'
  return None


def make_decompressor(compression):
  if compression == 'bz2':
    return bz2.BZ2Decompressor()
  if compression == 'zst':
    return zstd.ZstdDecompressor().decompressobj()
  raise ValueError(f"Unsupported compression type: {compression}")


def decompress_file(source, destination, compression=None):
  with open(source, 'rb') as src, open(destination, 'wb') as dst:
    header = src.read(4)
    compression = compression or compression_type(header)
    decompressor = make_decompressor(compression)
    dst.write(decompressor.decompress(header))
    while data := src.read(1024 * 1024):
      dst.write(decompressor.decompress(data))
  if not decompressor.eof:
    raise EOFError(f"Compressed {compression} file ended before the end-of-stream marker")


def materialize_cached_file(source, url, compression):
  local_path = cache_file_path(url, compression)
  if os.path.exists(local_path):
    return local_path

  tmp_fd, tmp_path = tempfile.mkstemp(dir=Paths.download_cache_root())
  os.close(tmp_fd)
  try:
    decompress_file(source, tmp_path, compression)
    shutil.move(tmp_path, local_path)
  except Exception:
    try:
      os.unlink(tmp_path)
    except OSError:
      pass
    raise
  return local_path


def cmd_route_files(args):
  api_call(lambda api: api.get(f"v1/route/{args.route}/files"))


def cmd_download(args):
  url = args.url
  use_cache = not args.no_cache

  if use_cache:
    for compression in ('bz2', 'zst'):
      decompressed_path = cache_file_path(url, compression)
      if os.path.exists(decompressed_path):
        sys.stdout.write(decompressed_path + "\n")
        sys.stdout.flush()
        return

    local_path = cache_file_path(url)
    if os.path.exists(local_path):
      with open(local_path, 'rb') as f:
        compression = compression_type(f.read(4))
        if compression:
          local_path = materialize_cached_file(local_path, url, compression)
      sys.stdout.write(local_path + "\n")
      sys.stdout.flush()
      return

  try:
    # Stream the file in a single HTTP request instead of making
    # a separate Range request per chunk (which was very slow).
    pool = URLFile.pool_manager()
    r = pool.request("GET", url, preload_content=False)
    if r.status not in (200, 206):
      sys.stderr.write(f"ERROR:HTTP {r.status}\n")
      sys.stderr.flush()
      sys.exit(1)

    total = int(r.headers.get('content-length', 0))
    if total <= 0:
      sys.stderr.write("ERROR:File not found or empty\n")
      sys.stderr.flush()
      sys.exit(1)

    os.makedirs(Paths.download_cache_root(), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=Paths.download_cache_root())
    try:
      downloaded = 0
      chunk_size = 1024 * 1024
      compression = None
      decompressor = None
      with os.fdopen(tmp_fd, 'wb') as f:
        for data in r.stream(chunk_size):
          if downloaded == 0:
            compression = compression_type(data)
            if compression:
              decompressor = make_decompressor(compression)
          f.write(decompressor.decompress(data) if decompressor else data)
          downloaded += len(data)
          sys.stderr.write(f"PROGRESS:{downloaded}:{total}\n")
          sys.stderr.flush()

      if decompressor and not decompressor.eof:
        raise EOFError(f"Compressed {compression} file ended before the end-of-stream marker")

      if decompressor:
        if use_cache:
          output_path = cache_file_path(url, compression)
          shutil.move(tmp_path, output_path)
        else:
          output_path = tmp_path
        sys.stdout.write(output_path + "\n")
      elif use_cache:
        shutil.move(tmp_path, local_path)
        sys.stdout.write(local_path + "\n")
      else:
        sys.stdout.write(tmp_path + "\n")
    except Exception:
      try:
        os.unlink(tmp_path)
      except OSError:
        pass
      raise
    finally:
      r.release_conn()

  except Exception as e:
    sys.stderr.write(f"ERROR:{e}\n")
    sys.stderr.flush()
    sys.exit(1)

  sys.stdout.flush()


def cmd_decompress(args):
  os.makedirs(Paths.download_cache_root(), exist_ok=True)
  output_fd, output_path = tempfile.mkstemp(dir=Paths.download_cache_root())
  os.close(output_fd)
  try:
    decompress_file(args.path, output_path)
  except Exception as e:
    try:
      os.unlink(output_path)
    except OSError:
      pass
    sys.stderr.write(f"ERROR:{e}\n")
    sys.stderr.flush()
    sys.exit(1)
  sys.stdout.write(output_path + "\n")
  sys.stdout.flush()


def cmd_devices(args):
  api_call(lambda api: api.get("v1/me/devices/"))


def cmd_device_routes(args):
  def fetch(api):
    if args.preserved:
      return api.get(f"v1/devices/{args.dongle_id}/routes/preserved")
    params = {}
    if args.start is not None:
      params['start'] = args.start
    if args.end is not None:
      params['end'] = args.end
    return api.get(f"v1/devices/{args.dongle_id}/routes_segments", params=params)
  api_call(fetch)


def main():
  parser = argparse.ArgumentParser(description="File downloader CLI for openpilot tools")
  subparsers = parser.add_subparsers(dest="command", required=True)

  p_rf = subparsers.add_parser("route-files")
  p_rf.add_argument("route")
  p_rf.set_defaults(func=cmd_route_files)

  p_dl = subparsers.add_parser("download")
  p_dl.add_argument("url")
  p_dl.add_argument("--no-cache", action="store_true")
  p_dl.set_defaults(func=cmd_download)

  p_dc = subparsers.add_parser("decompress")
  p_dc.add_argument("path")
  p_dc.set_defaults(func=cmd_decompress)

  p_dev = subparsers.add_parser("devices")
  p_dev.set_defaults(func=cmd_devices)

  p_dr = subparsers.add_parser("device-routes")
  p_dr.add_argument("dongle_id")
  p_dr.add_argument("--start", type=int, default=None)
  p_dr.add_argument("--end", type=int, default=None)
  p_dr.add_argument("--preserved", action="store_true")
  p_dr.set_defaults(func=cmd_device_routes)

  args = parser.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
