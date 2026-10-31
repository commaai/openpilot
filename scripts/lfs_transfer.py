#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub==1.7.1", "hf-xet==1.7.0"]
# ///
"""Git LFS standalone transfer agent for Hugging Face buckets."""

import argparse
import hashlib
import http.client
import importlib.util
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

try:
  from huggingface_hub import HfApi
except ImportError:
  API = None
else:
  API = HfApi() if importlib.util.find_spec("hf_xet") and hasattr(HfApi, "download_bucket_files") else None

BUCKET = os.environ.get("OPENPILOT_LFS_BUCKET", "commaai/openpilot-lfs")
TRUE_VALUES = ("1", "ON", "YES", "TRUE")


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


def send(message):
  print(json.dumps(message), flush=True)


def download_http(oid, size, path):
  request = urllib.request.Request(f"https://huggingface.co/buckets/{BUCKET}/resolve/sha256/{oid}")
  if os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "").upper() not in TRUE_VALUES:
    hf_home = Path(os.environ.get("HF_HOME", Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")) / "huggingface")).expanduser()
    token_path = Path(os.environ.get("HF_TOKEN_PATH", hf_home / "token")).expanduser()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and token_path.is_file():
      token = token_path.read_text().strip()
    if token:
      request.add_unredirected_header("Authorization", f"Bearer {token.strip()}")
  for attempt in range(4):
    try:
      with urllib.request.urlopen(request, timeout=30) as response, path.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
      if path.stat().st_size != size:
        raise http.client.HTTPException(f"incomplete download for {oid}")
      return
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError) as error:
      retryable = not isinstance(error, urllib.error.HTTPError) or error.code in (408, 429) or 500 <= error.code < 600
      if not retryable or attempt == 3:
        raise
      time.sleep(2 ** attempt)


def upload(oid, size, path):
  key = f"sha256/{oid}"
  existing = list(API.get_bucket_paths_info(BUCKET, [key]))
  if existing:
    if existing[0].size != size:
      raise RuntimeError(f"invalid bucket object {oid}: expected size {size}")
    return
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  if path.stat().st_size != size or digest.hexdigest() != oid:
    raise RuntimeError(f"invalid local LFS object {oid}")
  API.batch_bucket_files(BUCKET, add=[(path, key)])
  uploaded = list(API.get_bucket_paths_info(BUCKET, [key]))
  if not uploaded or uploaded[0].size != size:
    raise RuntimeError(f"uploaded object {oid} is missing or has the wrong size")


def transfer(upload_initialized=False):
  init = {"operation": "upload"} if upload_initialized else json.loads(sys.stdin.readline())
  if init["operation"] == "upload" and API is None:
    uv = shutil.which("uv")
    if not uv or upload_initialized:
      send({"error": {"code": 1, "message": "Install uv to upload LFS objects through Xet."}})
      return
    # Git LFS waits for the init response, so no transfer requests have been read yet.
    os.execv(uv, [uv, "run", "--script", str(Path(__file__).resolve()), "--upload-initialized"])
  send({})
  git_dir = git("rev-parse", "--path-format=absolute", "--git-common-dir")
  with tempfile.TemporaryDirectory(prefix="lfs-transfer-", dir=git_dir) as directory:
    for line in sys.stdin:
      request = json.loads(line)
      if request["event"] == "terminate":
        return
      oid, size = request["oid"], request["size"]
      response = {"event": "complete", "oid": oid}
      try:
        if request["event"] == "upload":
          upload(oid, size, Path(request["path"]))
        else:
          path = Path(directory) / oid
          if API is not None and os.environ.get("HF_HUB_DISABLE_XET", "").upper() not in TRUE_VALUES:
            API.download_bucket_files(BUCKET, [(f"sha256/{oid}", path)], raise_on_missing_files=True)
          else:
            download_http(oid, size, path)
          response["path"] = str(path)
        send({"event": "progress", "oid": oid, "bytesSoFar": size, "bytesSinceLast": size})
      except Exception as error:
        response["error"] = {"code": 1, "message": str(error)}
      send(response)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", nargs="?", choices=("install",))
  parser.add_argument("--upload-initialized", action="store_true", help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.command == "install":
    install()
  else:
    transfer(args.upload_initialized)
