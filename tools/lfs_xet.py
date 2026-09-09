#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["hf-xet==1.4.2"]
# ///
"""Git LFS download agent for the public openpilot Hugging Face repository."""
from functools import cache
import json
from pathlib import Path
import re
import sys
import tempfile
import urllib.request

LFS_URL = "https://huggingface.co/commaai/openpilot-lfs.git/info/lfs"
TOKEN_URL = "https://huggingface.co/api/models/commaai/openpilot-lfs/xet-read-token/main"


def request_json(url, data=None):
  body = json.dumps(data).encode() if data is not None else None
  request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
  with urllib.request.urlopen(request, timeout=30) as response:
    return json.load(response)


def send(message):
  print(json.dumps(message), flush=True)


@cache
def xet_connection():
  return request_json(TOKEN_URL)


def download(oid, size, path):
  obj = request_json(f"{LFS_URL}/objects/batch", {
    "operation": "download", "transfers": ["basic"], "objects": [{"oid": oid, "size": size}],
  })["objects"][0]
  if "error" in obj:
    raise RuntimeError(obj["error"]["message"])
  action = obj["actions"]["download"]
  transferred = 0

  def progress(amount):
    nonlocal transferred
    transferred += amount
    send({"event": "progress", "oid": oid, "bytesSoFar": transferred, "bytesSinceLast": amount})

  # The LFS bridge URL identifies the Xet file by hash, including objects uploaded without a Hub Git commit.
  xet_hash = re.search(r"/xet-bridge-[^/]+/[^/]+/([0-9a-f]{64})(?:\?|$)", action["href"])
  if xet_hash:
    from hf_xet import PyXetDownloadInfo, download_files

    def refresh_token():
      xet_connection.cache_clear()
      info = xet_connection()
      return info["accessToken"], info["exp"]

    info = xet_connection()
    download_files([PyXetDownloadInfo(str(path), xet_hash[1], size)], endpoint=info["casUrl"],
                   token_info=(info["accessToken"], info["exp"]), token_refresher=refresh_token, progress_updater=[progress])
  else:
    # Newly uploaded LFS objects may not have been converted to Xet yet.
    request = urllib.request.Request(action["href"], headers=action.get("header", {}))
    with urllib.request.urlopen(request, timeout=30) as response, path.open("wb") as output:
      while chunk := response.read(1024 * 1024):
        output.write(chunk)
        progress(len(chunk))


def main():
  with tempfile.TemporaryDirectory(prefix="git-lfs-xet-") as directory:
    for line in sys.stdin:
      message = json.loads(line)
      if message["event"] == "init":
        if message["operation"] != "download":
          send({"error": {"code": 1, "message": "This agent only supports downloads"}})
          return
        send({})
      elif message["event"] == "terminate":
        return
      elif message["event"] == "download":
        oid = message["oid"]
        path = Path(directory) / oid
        try:
          download(oid, message["size"], path)
          send({"event": "complete", "oid": oid, "path": str(path)})
        except Exception as e:
          path.unlink(missing_ok=True)
          # Exceptions from HTTP/native clients can contain signed URLs; keep those out of Git's logs.
          send({"event": "complete", "oid": oid, "error": {"code": 1, "message": f"Download failed ({type(e).__name__})"}})


if __name__ == "__main__":
  main()
