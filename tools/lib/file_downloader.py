#!/usr/bin/env python3
"""
CLI tool for downloading files and querying the comma API.
Called by C++ replay/cabana via subprocess.

Subcommands:
  route-files <route>    - Get route file URLs as JSON
  download <url>         - Download URL to local cache, print local path
  devices                - List user's devices as JSON
  device-routes <did>    - List routes for a device as JSON
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import shutil

from openpilot.system.hardware.hw import Paths
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


def cache_file_path(url):
  url_without_query = url.split("?")[0]
  return os.path.join(Paths.download_cache_root(), hashlib.sha256(url_without_query.encode()).hexdigest())


def cmd_route_files(args):
  api_call(lambda api: api.get(f"v1/route/{args.route}/files"))


def cmd_download(args):
  url = args.url
  use_cache = not args.no_cache

  if use_cache:
    local_path = cache_file_path(url)
    if os.path.exists(local_path):
      sys.stdout.write(local_path + "\n")
      sys.stdout.flush()
      return

  try:
    uf = URLFile(url, cache=False)
    total = uf.get_length()
    if total <= 0:
      sys.stderr.write("ERROR:File not found or empty\n")
      sys.stderr.flush()
      sys.exit(1)

    os.makedirs(Paths.download_cache_root(), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=Paths.download_cache_root())
    try:
      downloaded = 0
      chunk_size = 1024 * 1024
      with os.fdopen(tmp_fd, 'wb') as f:
        while downloaded < total:
          data = uf.read(min(chunk_size, total - downloaded))
          f.write(data)
          downloaded += len(data)
          sys.stderr.write(f"PROGRESS:{downloaded}:{total}\n")
          sys.stderr.flush()

      if use_cache:
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

  except Exception as e:
    sys.stderr.write(f"ERROR:{e}\n")
    sys.stderr.flush()
    sys.exit(1)

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
