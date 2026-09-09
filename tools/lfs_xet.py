#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub==1.7.1", "hf-xet==1.4.2"]
# ///
"""Git LFS download agent for the public openpilot Hugging Face repository."""
import json
from pathlib import Path
import re
import sys
import tempfile

from huggingface_hub.file_download import http_get, xet_get
from huggingface_hub.utils import XetFileData, get_session, logging, tqdm

LFS_URL = "https://huggingface.co/commaai/openpilot-lfs.git/info/lfs"
TOKEN_URL = "https://huggingface.co/api/models/commaai/openpilot-lfs/xet-read-token/main"
logging.set_verbosity_error()


def send(message):
  print(json.dumps(message), flush=True)


def download(oid, size, path):
  response = get_session().post(f"{LFS_URL}/objects/batch", timeout=30, json={
    "operation": "download", "transfers": ["basic"], "objects": [{"oid": oid, "size": size}],
  })
  response.raise_for_status()
  obj = response.json()["objects"][0]
  if "error" in obj:
    raise RuntimeError(obj["error"]["message"])
  action = obj["actions"]["download"]

  class Progress(tqdm):
    def __init__(self, **kwargs):
      super().__init__(initial=kwargs.get("initial", 0), disable=True)

    def update(self, amount):
      self.n += amount
      send({"event": "progress", "oid": oid, "bytesSoFar": self.n, "bytesSinceLast": amount})

  # The LFS bridge URL identifies the Xet file by hash, including objects uploaded without a Hub Git commit.
  xet_hash = re.search(r"/xet-bridge-[^/]+/[^/]+/([0-9a-f]{64})(?:\?|$)", action["href"])
  options = {"expected_size": size, "displayed_filename": oid, "tqdm_class": Progress}
  if xet_hash:
    xet_get(incomplete_path=path, xet_file_data=XetFileData(xet_hash[1], TOKEN_URL), headers={}, **options)
  else:
    # Newly uploaded LFS objects may not have been converted to Xet yet.
    with path.open("wb") as output:
      http_get(action["href"], output, headers=action.get("header", {}), **options)


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
