#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub==1.7.1", "hf-xet==1.7.0"]
# ///
"""A small Git LFS client backed by a Hugging Face bucket."""

import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import http.client
import importlib.util
import io
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

try:
  from huggingface_hub import CommitOperationAdd, HfApi
except ImportError:
  API = None
else:
  API = HfApi() if importlib.util.find_spec("hf_xet") and hasattr(HfApi, "download_bucket_files") else None

POINTER_VERSION = "https://git-lfs.github.com/spec/v1"
POINTER_RE = re.compile(rb"\Aversion " + POINTER_VERSION.encode() + rb"\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n\Z")
BUCKET = os.environ.get("OPENPILOT_LFS_BUCKET", "commaai/openpilot-lfs")
LEGACY_REPO = os.environ.get("OPENPILOT_LFS_LEGACY_REPO", "commaai/openpilot-lfs" if BUCKET == "commaai/openpilot-lfs" else "")
ZERO = "0" * 40
TRANSFERS = 4
TRUE_VALUES = ("1", "ON", "YES", "TRUE")


def git(*args, stdin=None, check=True):
  return subprocess.run(["git", *args], input=stdin, capture_output=True, check=check).stdout


ROOT, GIT_DIR = map(Path, git("rev-parse", "--path-format=absolute", "--show-toplevel", "--git-common-dir").decode().splitlines())
STORE = GIT_DIR / "lfs" / "objects"


def remove_git_lfs_hooks(directory):
  for name in ("pre-push", "post-checkout", "post-commit", "post-merge"):
    path = directory / name
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024:
      continue
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    commands = {f'git lfs {name} "$@"'}
    if name == "pre-push":
      commands.update(('git lfs push --stdin $*', 'git lfs push --stdin "$@"'))
      if lines and re.fullmatch(r'exec "[^"]*/git-lfs" pre-push "\$@"', lines[-1]):  # shim from tools/op.sh
        commands.add(lines[-1])
    has_lfs_guard = len(lines) == 3 and lines[1].startswith("command -v git-lfs ")
    if lines[:1] == ["#!/bin/sh"] and lines[-1] in commands and (len(lines) == 2 or has_lfs_guard):
      path.unlink()


def install(skip_custom_hooks=False):
  script = GIT_DIR / "lfs.py"
  hook = Path(git("rev-parse", "--path-format=absolute", "--git-path", "hooks/pre-push").decode().strip())
  remove_git_lfs_hooks(hook.parent)
  hook_script = '#!/bin/sh\nexec "$(git rev-parse --git-common-dir)/lfs.py" pre-push "$@"\n'
  installed_hook = f'#!/bin/sh\nexec {shlex.quote(str(script))} pre-push "$@"\n'
  previous_hook = '#!/bin/sh\nexec "$(git rev-parse --show-toplevel)/lfs.py" pre-push "$@"\n'
  if hook.is_file() and hook.read_text(errors="replace") not in (hook_script, installed_hook, previous_hook):
    if not skip_custom_hooks:
      raise RuntimeError(f"refusing to replace custom hook {hook}")
    hook = None
  if Path(__file__).resolve() != script:
    shutil.copyfile(__file__, script)
  script.chmod(0o755)
  if hook is not None:
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(hook_script)
    hook.chmod(0o755)
  git("config", "--worktree", "--remove-section", "filter.lfs", check=False)
  git("config", "--local", "filter.lfs.process", '"$(git rev-parse --git-common-dir)/lfs.py" filter-process')
  git("config", "--local", "filter.lfs.required", "true")


def object_path(oid):
  return STORE / oid[:2] / oid[2:4] / oid


def pointer(oid, size):
  return f"version {POINTER_VERSION}\noid sha256:{oid}\nsize {size}\n".encode()


def parse_pointer(data):
  match = POINTER_RE.match(data)
  return (match[1].decode(), int(match[2])) if match else None


@contextlib.contextmanager
def object_lock(oid):
  path = GIT_DIR / "lfs" / "locks" / (oid + ".lock")
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a+b") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    yield


def valid(path, size):
  return path.is_file() and path.stat().st_size == size


def hash_file(path):
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def store(chunks):
  STORE.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="incoming-", dir=STORE) as directory:
    tmp = Path(directory) / "object"
    digest, size = hashlib.sha256(), 0
    with tmp.open("xb") as out:
      for chunk in chunks:
        out.write(chunk)
        digest.update(chunk)
        size += len(chunk)
    if size == 0:
      return b""
    if size <= 1024:
      data = tmp.read_bytes()
      if parse_pointer(data):
        return data
    oid = digest.hexdigest()
    dst = object_path(oid)
    with object_lock(oid):
      if not valid(dst, size):
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, dst)
    return pointer(oid, size)


def parallel(function, arguments, workers=TRANSFERS):
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    futures = [pool.submit(function, *args) for args in arguments]
    return [future.result() for future in futures]


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


def download_one(oid, size):
  dst = object_path(oid)
  with object_lock(oid):
    if valid(dst, size):
      return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{oid}-", dir=dst.parent) as directory:
      tmp = Path(directory) / "object"
      if API is not None and os.environ.get("HF_HUB_DISABLE_XET", "").upper() not in TRUE_VALUES:
        API.download_bucket_files(BUCKET, [(f"sha256/{oid}", tmp)], raise_on_missing_files=True)
      else:
        download_http(oid, size, tmp)
      if tmp.stat().st_size != size or hash_file(tmp) != oid:
        raise RuntimeError(f"sha256 mismatch for {oid}")
      os.replace(tmp, dst)
      return dst


def smudge(data, name, exclude):
  if os.environ.get("GIT_LFS_SKIP_SMUDGE") == "1" or name == exclude:
    return data
  parsed = parse_pointer(data)
  if not parsed:
    return data
  oid, size = parsed
  path = object_path(oid)
  if not valid(path, size):
    download_one(oid, size)
  return path


def read_packets(stream):
  while True:
    header = stream.read(4)
    if len(header) != 4:
      raise EOFError
    length = int(header, 16)
    if length == 0:
      return
    if length < 4:
      raise RuntimeError("unsupported pkt-line control packet")
    yield stream.read(length - 4)


def write_packets(source):
  if isinstance(source, bytes):
    source = (source,)
  for chunk in source:
    for start in range(0, len(chunk), 65512):
      packet = chunk[start : start + 65512]
      sys.stdout.buffer.write(f"{len(packet) + 4:04x}".encode() + packet)
  sys.stdout.buffer.write(b"0000")
  sys.stdout.buffer.flush()


def filter_process():
  exclude = git("config", "--get", "lfs.fetchexclude", check=False).decode().strip()
  list(read_packets(sys.stdin.buffer))
  write_packets((b"git-filter-server\n", b"version=2\n"))
  list(read_packets(sys.stdin.buffer))
  write_packets((b"capability=clean\n", b"capability=smudge\n"))
  while True:
    try:
      fields = dict(line.rstrip(b"\n").split(b"=", 1) for line in read_packets(sys.stdin.buffer))
    except EOFError:
      return
    content = read_packets(sys.stdin.buffer)
    result = store(content) if fields[b"command"] == b"clean" else smudge(b"".join(content), os.fsdecode(fields[b"pathname"]), exclude)
    write_packets(b"status=success\n")
    if isinstance(result, Path):
      with result.open("rb") as source_stream:
        write_packets(iter(lambda: source_stream.read(65512), b""))
    else:
      write_packets(result)
    write_packets(())


def cat_small(oids):
  checked = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)", stdin=("\n".join(oids) + "\n").encode()).decode().splitlines()
  small = [line.split()[0] for line in checked if len(line.split()) == 3 and line.split()[1] == "blob" and int(line.split()[2]) <= 1024]
  stream = io.BytesIO(git("cat-file", "--batch", stdin=("\n".join(small) + "\n").encode()))
  for oid in small:
    size = int(stream.readline().split()[2])
    yield oid, stream.read(size)
    stream.read(1)


def pointers_for_index():
  paths = {}
  for entry in filter(None, git("-C", str(ROOT), "ls-files", "--stage", "-z").split(b"\0")):
    metadata, path = entry.split(b"\t", 1)
    mode, oid, stage = metadata.split()
    if stage == b"0" and mode in (b"100644", b"100755"):
      paths.setdefault(oid.decode(), []).append(path.decode(errors="surrogateescape"))
  found = {}
  for oid, data in cat_small(paths):
    if parsed := parse_pointer(data):
      found.setdefault(parsed, []).extend(paths[oid])
  return found


def pull(exclude=None):
  install()
  found = pointers_for_index()
  if exclude is None:
    exclude = git("config", "--get", "lfs.fetchexclude", check=False).decode().strip()
  if exclude:
    found = {obj: kept for obj, paths in found.items() if (kept := [path for path in paths if path != exclude])}
  parallel(download_one, found)
  for (oid, size), paths in found.items():
    for name in paths:
      relative = Path(name)
      if any((ROOT / part).is_symlink() for part in (relative, *relative.parents)):
        continue
      path = ROOT / relative
      try:
        if path.stat().st_size > 1024 or parse_pointer(path.read_bytes()) != (oid, size):
          continue
      except (FileNotFoundError, IsADirectoryError):
        continue
      with tempfile.TemporaryDirectory(prefix=path.name + "-", dir=path.parent) as directory:
        tmp = shutil.copyfile(object_path(oid), Path(directory) / "object")
        shutil.copymode(path, tmp)
        Path(tmp).replace(path)


def upload(objects):
  objects = dict(objects)
  if not objects:
    return
  existing = {f.path: f.size for f in API.get_bucket_paths_info(BUCKET, (f"sha256/{oid}" for oid in objects))}
  pending = []
  additions = []
  for oid, size in objects.items():
    key = f"sha256/{oid}"
    if key in existing:
      if existing[key] != size:
        raise RuntimeError(f"invalid bucket object {oid}: size {existing[key]}, expected {size}")
      if not LEGACY_REPO:
        continue
    path = download_one(oid, size) if key in existing else object_path(oid)
    if not valid(path, size):
      raise RuntimeError(f"missing local LFS object {oid}")
    if hash_file(path) != oid:
      raise RuntimeError(f"local sha256 mismatch for {oid}")
    if LEGACY_REPO:
      addition = CommitOperationAdd(path_in_repo=key, path_or_fileobj=path)
      # These are LFS objects regardless of the legacy repository's filename rules.
      addition._upload_mode = "lfs"
      additions.append(addition)
    if key not in existing:
      pending.append((path, key))
  if additions:
    API.preupload_lfs_files(LEGACY_REPO, additions, free_memory=False)
  if pending:
    API.batch_bucket_files(BUCKET, add=pending)
    uploaded = {f.path: f.size for f in API.get_bucket_paths_info(BUCKET, (key for _path, key in pending))}
    for path, key in pending:
      if uploaded.get(key) != objects[path.name]:
        raise RuntimeError(f"uploaded object {path.name} is missing or has the wrong size")


def pre_push(remote):
  if os.environ.get("GIT_LFS_SKIP_PUSH") == "1":
    return
  input_data = sys.stdin.buffer.read()
  updates = (line.decode().split() for line in input_data.splitlines())
  local_oids = [local_oid for _local_ref, local_oid, _remote_ref, _remote_oid in updates if local_oid != ZERO]
  if not local_oids:
    return
  output = git("rev-list", "--objects", "--no-object-names", *local_oids, "--not", f"--remotes={remote}")
  object_ids = output.decode().splitlines()
  objects = [parsed for _oid, data in cat_small(object_ids) if (parsed := parse_pointer(data))]
  if objects and API is None:
    if not shutil.which("uv"):
      raise RuntimeError("Install uv to upload LFS objects through Xet.")
    subprocess.run(["uv", "run", "--script", str(Path(__file__).resolve()), "pre-push", remote], input=input_data, check=True)
  else:
    upload(objects)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", choices=("filter-process", "pull", "pre-push", "install"))
  parser.add_argument("remote", nargs="?", default="origin")
  parser.add_argument("url", nargs="?")
  parser.add_argument("--exclude", help="path to leave as a pointer when pulling; overrides lfs.fetchexclude")
  parser.add_argument("--skip-custom-hooks", action="store_true", help="keep custom pre-push hooks when installing")
  args = parser.parse_args()
  if args.command == "filter-process":
    filter_process()
  elif args.command == "pull":
    pull(args.exclude)
  elif args.command == "pre-push":
    pre_push(args.remote)
  elif args.command == "install":
    install(args.skip_custom_hooks)


if __name__ == "__main__":
  main()
