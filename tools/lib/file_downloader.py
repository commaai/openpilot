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


def get_api():
  token = get_token()
  return CommaApi(token)


def cache_file_path(url):
  """Cache key: sha256 of URL without query string (matches C++ cacheFilePath convention)."""
  url_without_query = url.split("?")[0]
  return os.path.join(Paths.download_cache_root(), hashlib.sha256(url_without_query.encode()).hexdigest())


def cmd_route_files(args):
  try:
    api = get_api()
    result = api.get(f"v1/route/{args.route}/files")
    json.dump(result, sys.stdout)
  except UnauthorizedError:
    json.dump({"error": "unauthorized"}, sys.stdout)
  except APIError as e:
    if hasattr(e, 'status_code') and e.status_code == 404:
      json.dump({"error": "not_found"}, sys.stdout)
    else:
      json.dump({"error": str(e)}, sys.stdout)
  except Exception as e:
    json.dump({"error": str(e)}, sys.stdout)
  sys.stdout.write("\n")
  sys.stdout.flush()


def cmd_download(args):
  url = args.url
  use_cache = not args.no_cache
  local_path = cache_file_path(url)

  # Check cache first
  if use_cache and os.path.exists(local_path):
    sys.stdout.write(local_path + "\n")
    sys.stdout.flush()
    return

  try:
    # Get file size for progress
    uf = URLFile(url, cache=False)
    total = uf.get_length()
    if total <= 0:
      sys.stderr.write("ERROR:File not found or empty\n")
      sys.stderr.flush()
      sys.exit(1)

    # Download in chunks, report progress on stderr
    os.makedirs(Paths.download_cache_root(), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=Paths.download_cache_root())
    try:
      downloaded = 0
      chunk_size = 1024 * 1024  # 1MB
      with os.fdopen(tmp_fd, 'wb') as f:
        while downloaded < total:
          to_read = min(chunk_size, total - downloaded)
          uf.seek(downloaded)
          data = uf.read_aux(to_read)
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
      # Clean up temp file on error
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
  try:
    api = get_api()
    result = api.get("v1/me/devices/")
    json.dump(result, sys.stdout)
  except UnauthorizedError:
    json.dump({"error": "unauthorized"}, sys.stdout)
  except Exception as e:
    json.dump({"error": str(e)}, sys.stdout)
  sys.stdout.write("\n")
  sys.stdout.flush()


def cmd_device_routes(args):
  try:
    api = get_api()
    if args.preserved:
      endpoint = f"v1/devices/{args.dongle_id}/routes/preserved"
      result = api.get(endpoint)
    else:
      params = {}
      if args.start is not None:
        params['start'] = args.start
      if args.end is not None:
        params['end'] = args.end
      endpoint = f"v1/devices/{args.dongle_id}/routes_segments"
      result = api.get(endpoint, params=params)
    json.dump(result, sys.stdout)
  except UnauthorizedError:
    json.dump({"error": "unauthorized"}, sys.stdout)
  except Exception as e:
    json.dump({"error": str(e)}, sys.stdout)
  sys.stdout.write("\n")
  sys.stdout.flush()


def main():
  parser = argparse.ArgumentParser(description="File downloader CLI for openpilot tools")
  subparsers = parser.add_subparsers(dest="command", required=True)

  # route-files
  p_rf = subparsers.add_parser("route-files", help="Get route file URLs")
  p_rf.add_argument("route", help="Route string (dongle_id|timestamp)")
  p_rf.set_defaults(func=cmd_route_files)

  # download
  p_dl = subparsers.add_parser("download", help="Download URL to local cache")
  p_dl.add_argument("url", help="URL to download")
  p_dl.add_argument("--no-cache", action="store_true", help="Skip cache")
  p_dl.set_defaults(func=cmd_download)

  # devices
  p_dev = subparsers.add_parser("devices", help="List user devices")
  p_dev.set_defaults(func=cmd_devices)

  # device-routes
  p_dr = subparsers.add_parser("device-routes", help="List routes for a device")
  p_dr.add_argument("dongle_id", help="Device dongle ID")
  p_dr.add_argument("--start", type=int, default=None, help="Start time in ms")
  p_dr.add_argument("--end", type=int, default=None, help="End time in ms")
  p_dr.add_argument("--preserved", action="store_true", help="List preserved routes")
  p_dr.set_defaults(func=cmd_device_routes)

  args = parser.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
