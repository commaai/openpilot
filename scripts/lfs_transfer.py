#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub==1.7.1", "hf-xet==1.7.0"]
# ///
"""Git LFS standalone transfer agent for Hugging Face buckets."""

import argparse
from contextlib import nullcontext
import functools
import hashlib
import http.client
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

BUCKET = os.environ.get("OPENPILOT_LFS_BUCKET", "commaai/openpilot-lfs")
TRUE_VALUES = ("1", "ON", "YES", "TRUE")
RANGES = 8
RANGE_MIN = 16 << 20


@functools.cache
def get_api():
  try:
    from huggingface_hub import HfApi
  except ImportError:
    return None
  return HfApi() if importlib.util.find_spec("hf_xet") and hasattr(HfApi, "batch_bucket_files") else None


def git(*args):
  return subprocess.check_output(["git", *args]).decode().strip()


def install():
  script = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir")) / "lfs_transfer.py"
  if Path(__file__).resolve() != script:
    shutil.copyfile(__file__, script)
  script.chmod(0o755)
  git("config", "--local", "lfs.customtransfer.hf-bucket.path", "sh")
  git("config", "--local", "lfs.customtransfer.hf-bucket.args", '-c \'exec "$(git rev-parse --git-common-dir)/lfs_transfer.py"\'')
  git("config", "--local", "lfs.standalonetransferagent", "hf-bucket")
  git("config", "--local", "lfs.locksverify", "false")
  transfers = git("config", "--get", "--default", "16", "lfs.concurrenttransfers")
  git("config", "--local", "lfs.concurrenttransfers", transfers)


def send(message):
  print(json.dumps(message), flush=True)


def hf_request(url, **kwargs):
  request = urllib.request.Request(url, **kwargs)
  if os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "").upper() not in TRUE_VALUES:
    hf_home = Path(os.environ.get("HF_HOME", Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")) / "huggingface")).expanduser()
    token_path = Path(os.environ.get("HF_TOKEN_PATH", hf_home / "token")).expanduser()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and token_path.is_file():
      token = token_path.read_text().strip()
    if token:
      request.add_unredirected_header("Authorization", f"Bearer {token.strip()}")
  return request


def object_exists(oid, size):
  request = hf_request(f"https://huggingface.co/api/buckets/{BUCKET}/paths-info",
                       data=json.dumps({"paths": [f"sha256/{oid}"]}).encode(), headers={"Content-Type": "application/json"})
  with urllib.request.urlopen(request, timeout=30) as response:
    existing = json.load(response)
  if existing and existing[0]["size"] != size:
    raise RuntimeError(f"invalid bucket object {oid}: expected size {size}")
  return bool(existing)


def download_http(oid, size, path, span=None):
  ranges = min(RANGES, max(1, (size + RANGE_MIN - 1) // RANGE_MIN))
  if span is None and ranges > 1:
    from concurrent.futures import ThreadPoolExecutor

    with tempfile.TemporaryDirectory(dir=path.parent) as directory, ThreadPoolExecutor(max_workers=ranges) as pool:
      parts = [Path(directory) / str(i) for i in range(ranges)]
      futures = [pool.submit(download_http, oid, size, part, (i * size // ranges, (i + 1) * size // ranges - 1))
                 for i, part in enumerate(parts)]
      for future in futures:
        future.result()
      with path.open("wb") as out:
        for part in parts:
          with part.open("rb") as stream:
            shutil.copyfileobj(stream, out, length=1024 * 1024)
    return

  request = hf_request(f"https://huggingface.co/buckets/{BUCKET}/resolve/sha256/{oid}")
  if span is not None:
    request.add_header("Range", f"bytes={span[0]}-{span[1]}")
  expected_size = size if span is None else span[1] - span[0] + 1
  for attempt in range(4):
    try:
      with urllib.request.urlopen(request, timeout=30) as response, path.open("wb") as out:
        if span is not None and (response.status != 206 or response.headers.get("Content-Range") != f"bytes {span[0]}-{span[1]}/{size}"):
          raise RuntimeError(f"invalid range response for {oid}")
        shutil.copyfileobj(response, out, length=1024 * 1024)
      if path.stat().st_size != expected_size:
        raise http.client.HTTPException(f"incomplete download for {oid}")
      return
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError) as error:
      retryable = not isinstance(error, urllib.error.HTTPError) or error.code in (408, 429) or 500 <= error.code < 600
      if not retryable or attempt == 3:
        raise
      time.sleep(2 ** attempt)


def upload(oid, size, path, bootstrapped=False):
  if object_exists(oid, size):
    return
  api = get_api()
  if api is None:
    uv = shutil.which("uv")
    if not uv or bootstrapped:
      raise RuntimeError("Install uv to upload new LFS objects through Xet.")
    request = json.dumps({"event": "upload", "oid": oid, "size": size, "path": str(path)})
    # Git LFS waits for this object's completion before sending another request.
    os.execv(uv, [uv, "run", "--script", str(Path(__file__).resolve()), "--upload-request", request])
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  if path.stat().st_size != size or digest.hexdigest() != oid:
    raise RuntimeError(f"invalid local LFS object {oid}")
  api.batch_bucket_files(BUCKET, add=[(path, f"sha256/{oid}")])
  if not object_exists(oid, size):
    raise RuntimeError(f"uploaded object {oid} is missing or has the wrong size")


def transfer(upload_request=None):
  init = {"operation": "upload"} if upload_request else json.loads(sys.stdin.readline())
  if upload_request is None:
    send({})
  git_dir = git("rev-parse", "--path-format=absolute", "--git-common-dir")
  directory = tempfile.TemporaryDirectory(prefix="lfs-transfer-", dir=git_dir) if init["operation"] == "download" else nullcontext()
  with directory as directory:
    for line in itertools.chain([upload_request] if upload_request else [], sys.stdin):
      request = json.loads(line)
      if request["event"] == "terminate":
        return
      oid, size = request["oid"], request["size"]
      response = {"event": "complete", "oid": oid}
      try:
        if request["event"] == "upload":
          upload(oid, size, Path(request["path"]), bootstrapped=upload_request is not None)
        else:
          path = Path(directory) / oid
          download_http(oid, size, path)
          response["path"] = str(path)
        send({"event": "progress", "oid": oid, "bytesSoFar": size, "bytesSinceLast": size})
      except Exception as error:
        response["error"] = {"code": 1, "message": str(error)}
      send(response)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", nargs="?", choices=("install",))
  parser.add_argument("--upload-request", help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.command == "install":
    install()
  else:
    transfer(args.upload_request)
