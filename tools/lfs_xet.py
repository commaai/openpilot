#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub==1.7.1", "hf-xet==1.4.2"]
# ///
"""Git LFS download agent for the public openpilot Hugging Face repository."""
import json
import re
import sys
import tempfile
from pathlib import Path

from huggingface_hub.file_download import http_get, xet_get
from huggingface_hub.utils import XetFileData, get_session, logging, tqdm

LFS_URL = "https://huggingface.co/commaai/openpilot-lfs.git/info/lfs"
TOKEN_URL = "https://huggingface.co/api/models/commaai/openpilot-lfs/xet-read-token/main"


def send(**message):
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
    def __init__(self, initial=0, **kwargs):
      super().__init__(initial=initial, disable=True)

    def update(self, n=1):
      self.n += n
      send(event="progress", oid=oid, bytesSoFar=self.n, bytesSinceLast=n)

  # The LFS bridge URL exposes the Xet hash, even for objects without a Hub commit.
  xet_hash = re.search(r"/xet-bridge-[^/]+/[^/]+/([0-9a-f]{64})(?:\?|$)", action["href"])
  options = {"expected_size": size, "displayed_filename": oid, "tqdm_class": Progress}
  if xet_hash:
    xet_get(incomplete_path=path, xet_file_data=XetFileData(xet_hash[1], TOKEN_URL), headers={}, **options)
  else:
    # Newly uploaded LFS objects may not have been converted to Xet yet.
    with path.open("wb") as output:
      http_get(action["href"], output, headers=action.get("header", {}), **options)


if __name__ == "__main__":
  logging.set_verbosity_error()
  with tempfile.TemporaryDirectory(prefix="git-lfs-xet-") as directory:
    for line in sys.stdin:
      message = json.loads(line)
      match message["event"]:
        case "init":
          if message["operation"] != "download":
            send(error={"code": 1, "message": "This agent only supports downloads"})
            break
          send()
        case "terminate":
          break
        case "download":
          oid = message["oid"]
          path = Path(directory) / oid
          try:
            download(oid, message["size"], path)
            send(event="complete", oid=oid, path=str(path))
          except Exception as e:
            path.unlink(missing_ok=True)
            # HTTP/native errors can contain signed URLs; keep those out of Git's logs.
            send(event="complete", oid=oid, error={"code": 1, "message": f"Download failed ({type(e).__name__})"})
