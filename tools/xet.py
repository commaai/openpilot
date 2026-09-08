#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub==1.30.0", "hf-xet==1.6.0"]
# ///
"""Git LFS standalone transfer agent backed by Hugging Face Xet storage."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

REPO_ID = "commaai/openpilot-lfs"


def git(*args):
  return subprocess.check_output(["git", *args], text=True).strip()


def install():
  uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
  subprocess.run([uv, "run", "--script", "--locked", __file__, "configure"], check=True)


def configure():
  # Keep the adapter outside the worktree so checkouts of older commits work.
  # Normal clones use a relative .git path, preserving copied device checkouts.
  agent = Path(git("rev-parse", "--git-common-dir")) / "xet/transfer.py"
  agent.parent.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(__file__, agent)
  git("config", "--local", "lfs.customtransfer.xet.path", sys.executable)
  git("config", "--local", "lfs.customtransfer.xet.args", shlex.join([str(agent), "transfer"]))
  git("config", "--local", "lfs.customtransfer.xet.concurrent", "false")
  git("config", "--local", "lfs.standalonetransferagent", "xet")


def verify(path, oid, size):
  with path.open("rb") as f:
    if os.fstat(f.fileno()).st_size != size or hashlib.file_digest(f, "sha256").hexdigest() != oid:
      raise ValueError(f"size or SHA-256 mismatch for {oid}")


class XetTransfer:
  def __init__(self):
    from huggingface_hub import HfApi
    from huggingface_hub.utils._runtime import is_xet_available

    if not is_xet_available():
      raise RuntimeError("Xet is disabled or unavailable. Unset HF_HUB_DISABLE_XET and run python3 tools/xet.py install.")
    config = subprocess.run(["git", "config", "--get", "xet.repo"], text=True, capture_output=True)
    self.repo_id = config.stdout.strip() or REPO_ID
    self.api = HfApi()

  def upload(self, oid, size, path):
    verify(path, oid, size)
    filename = f"sha256/{oid}"
    existing = self.api.get_paths_info(self.repo_id, [filename])
    if existing:
      obj = existing[0]
      if obj.size != size or obj.lfs is None or obj.lfs.sha256 != oid:
        raise ValueError(f"remote object does not match {oid}")
      return
    self.api.upload_file(repo_id=self.repo_id, path_in_repo=filename, path_or_fileobj=path, commit_message=f"Add {oid}")

  def download(self, oid, size):
    from huggingface_hub import hf_hub_download

    cached = Path(hf_hub_download(repo_id=self.repo_id, filename=f"sha256/{oid}"))
    verify(cached, oid, size)
    # git-lfs moves the returned file into its object store. Never give it the
    # Hub cache entry itself, which may be shared with another process.
    temporary = Path(git("rev-parse", "--git-path", "lfs/tmp"))
    temporary.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=temporary, prefix="xet-", delete=False) as f:
      path = Path(f.name)
      try:
        with cached.open("rb") as source:
          shutil.copyfileobj(source, f)
      except BaseException:
        path.unlink(missing_ok=True)
        raise
    return str(path)


def transfer(input_stream=sys.stdin, output_stream=sys.stdout):
  client = None
  operation = None
  for line in input_stream:
    request = json.loads(line)
    event = request["event"]
    if event == "terminate":
      return
    response = {} if event == "init" else {"event": "complete", "oid": request.get("oid", "")}
    try:
      # The transfer protocol owns stdout, including during library imports.
      with contextlib.redirect_stdout(sys.stderr):
        if event == "init":
          operation = request["operation"]
          if operation not in ("upload", "download"):
            raise ValueError(f"unsupported operation: {operation}")
          client = XetTransfer()
        else:
          oid, size = request["oid"], request["size"]
          if client is None or event != operation:
            raise ValueError("transfer does not match initialization")
          if not re.fullmatch(r"[0-9a-f]{64}", oid) or not isinstance(size, int) or size < 0:
            raise ValueError("invalid LFS object")
          if event == "upload":
            client.upload(oid, size, Path(request["path"]))
          else:
            response["path"] = client.download(oid, size)
    except Exception as e:
      response["error"] = {"code": 1, "message": str(e)}
    print(json.dumps(response), file=output_stream, flush=True)


if __name__ == "__main__":
  match sys.argv[1:]:
    case ["install"]:
      install()
    case ["configure"]:
      configure()
    case ["transfer"]:
      transfer()
    case _:
      sys.exit("usage: python3 tools/xet.py {install|configure|transfer}")
