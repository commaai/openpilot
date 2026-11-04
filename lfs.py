#!/usr/bin/env python3
"""A small Git LFS client backed by a Hugging Face bucket."""

import argparse
import concurrent.futures
import contextlib
import fcntl
import fnmatch
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

POINTER_VERSION = "https://git-lfs.github.com/spec/v1"
POINTER_RE = re.compile(rb"\Aversion " + POINTER_VERSION.encode() + rb"\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n\Z")
BUCKET = os.environ.get("OPENPILOT_LFS_BUCKET", "commaai/openpilot-lfs")
ZERO = "0" * 40
RANGES = 8
RANGE_MIN = 16 << 20
TRUE_VALUES = ("1", "ON", "YES", "TRUE")


def git(*args, stdin=None, check=True):
  return subprocess.run(["git", *args], input=stdin, capture_output=True, check=check).stdout


ROOT, GIT_DIR = map(Path, git("rev-parse", "--path-format=absolute", "--show-toplevel", "--git-common-dir").decode().splitlines())
STORE = GIT_DIR / "lfs" / "objects"


# TODO: Remove this helper and its call once migration from stock Git LFS is no longer supported.
def remove_git_lfs_hooks(directory):
  for name in ("pre-push", "post-checkout", "post-commit", "post-merge"):
    path = directory / name
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024:
      continue
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    commands = {f'git lfs {name} "$@"'}
    if name == "pre-push":
      commands.update(('git lfs push --stdin $*', 'git lfs push --stdin "$@"'))
      if lines and re.fullmatch(r'exec "?[^"]*/git-lfs"? pre-push "\$@"', lines[-1]):  # hooks from tools/op.sh (#38775, #39114)
        commands.add(lines[-1])
    has_lfs_guard = len(lines) == 3 and lines[1].startswith("command -v git-lfs ")
    if lines[:1] == ["#!/bin/sh"] and lines[-1] in commands and (len(lines) == 2 or has_lfs_guard):
      path.unlink()


def write_executable(path, data):
  # replace atomically so git never runs a partly written script
  with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as tmp:
    tmp.write(data)
  os.chmod(tmp.name, 0o755)
  os.replace(tmp.name, path)


def install():
  script = GIT_DIR / "lfs.py"
  hook = Path(git("rev-parse", "--path-format=absolute", "--git-path", "hooks/pre-push").decode().strip())
  remove_git_lfs_hooks(hook.parent)
  remove_git_lfs_hooks(hook.parent / "post-commit.d")  # where `op post-commit` moves an existing post-commit hook
  hook_script = '#!/bin/sh\nexec "$(git rev-parse --git-common-dir)/lfs.py" pre-push "$@"\n'
  custom_hook = hook.is_file() and hook.read_text(errors="replace") != hook_script
  if Path(__file__).resolve() != script:
    write_executable(script, Path(__file__).read_bytes())
  script.chmod(0o755)
  if custom_hook:
    # keep it so pulls and updates still work; pushes won't upload LFS objects until the hook runs lfs.py
    print(f"lfs.py: keeping custom hook {hook}; add this line to it to upload LFS objects on push: {hook_script.splitlines()[-1]}", file=sys.stderr)
  else:
    hook.parent.mkdir(parents=True, exist_ok=True)
    write_executable(hook, hook_script.encode())
  # TODO: Remove legacy worktree filter cleanup once migration from stock Git LFS is no longer supported.
  git("config", "--worktree", "--remove-section", "filter.lfs", check=False)
  git("config", "--local", "filter.lfs.process", '"$(git rev-parse --git-common-dir)/lfs.py" filter-process')
  # a client that can't run fails git rather than committing raw files; failed downloads still leave the pointer
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
      else:
        dst.touch()
    return pointer(oid, size)


def transfer_workers():
  return int(git("config", "--get", "--default", "16", "lfs.concurrenttransfers"))


def parallel(function, arguments, workers=None):
  if workers is None:
    workers = transfer_workers()
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    futures = [pool.submit(function, *args) for args in arguments]
    return [future.result() for future in futures]


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


def object_info(objects):
  keys = [f"sha256/{oid}" for oid in objects]
  found = {}
  for start in range(0, len(keys), 1000):
    request = hf_request(f"https://huggingface.co/api/buckets/{BUCKET}/paths-info",
                         data=json.dumps({"paths": keys[start:start + 1000]}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
      found.update((item["path"], item["size"]) for item in json.load(response))
  return found


def download_http(oid, size, path, span=None):
  ranges = min(RANGES, max(1, (size + RANGE_MIN - 1) // RANGE_MIN))
  if span is None and ranges > 1:
    # each range writes its own slice of one preallocated file
    with path.open("wb") as out:
      out.truncate(size)
    parallel(download_http, [(oid, size, path, (i * size // ranges, (i + 1) * size // ranges - 1)) for i in range(ranges)], ranges)
    return

  request = hf_request(f"https://huggingface.co/buckets/{BUCKET}/resolve/sha256/{oid}")
  if span is not None:
    request.add_header("Range", f"bytes={span[0]}-{span[1]}")
  expected_size = size if span is None else span[1] - span[0] + 1
  for attempt in range(4):
    try:
      with urllib.request.urlopen(request, timeout=30) as response, path.open("wb" if span is None else "r+b") as out:
        if span is not None and (response.status != 206 or response.headers.get("Content-Range") != f"bytes {span[0]}-{span[1]}/{size}"):
          raise RuntimeError(f"invalid range response for {oid}")
        out.seek(0 if span is None else span[0])
        written = 0
        while chunk := response.read(1024 * 1024):
          out.write(chunk)
          written += len(chunk)
      if written != expected_size:
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
      dst.touch()
      return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{oid}-", dir=dst.parent) as directory:
      tmp = Path(directory) / "object"
      download_http(oid, size, tmp)
      if tmp.stat().st_size != size or hash_file(tmp) != oid:
        raise RuntimeError(f"sha256 mismatch for {oid}")
      os.replace(tmp, dst)
      return dst


def excluded(name, patterns):
  for pattern in filter(None, patterns.split(",")):
    pattern = pattern.strip().lstrip("/").rstrip("/")
    paths = [name, *(str(p) for p in Path(name).parents)]
    if "/" not in pattern:
      paths.extend(Path(name).parts)
    if pattern and any(fnmatch.fnmatchcase(path, pattern) for path in paths):
      return True
  return False


def smudge(data, name, exclude, pool=None):
  if os.environ.get("GIT_LFS_SKIP_SMUDGE", "").upper() in TRUE_VALUES or excluded(name, exclude):
    return data
  parsed = parse_pointer(data)
  if not parsed:
    return data
  oid, size = parsed
  if pool is not None and not valid(object_path(oid), size):
    return pool.submit(download_one, oid, size)
  return download_one(oid, size)


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
  capabilities = {line.rstrip(b"\n") for line in read_packets(sys.stdin.buffer)}
  write_packets(capability + b"\n" for capability in (b"capability=clean", b"capability=smudge", b"capability=delay")
                if capability in capabilities)
  pending, ready, pointers = {}, {}, {}
  with concurrent.futures.ThreadPoolExecutor(max_workers=transfer_workers()) as pool:
    while True:
      try:
        fields = dict(line.rstrip(b"\n").split(b"=", 1) for line in read_packets(sys.stdin.buffer))
      except EOFError:
        return
      if fields[b"command"] == b"list_available_blobs":
        if pending:
          concurrent.futures.wait(pending.values(), return_when=concurrent.futures.FIRST_COMPLETED)
        available = [name for name, future in pending.items() if future.done()]
        for name in available:
          ready[name] = pending.pop(name)
        write_packets(b"pathname=" + name + b"\n" for name in available)
        write_packets(b"status=success\n")
        continue
      content = read_packets(sys.stdin.buffer)
      name = fields[b"pathname"]
      if fields[b"command"] == b"clean":
        result = store(content)
      else:
        data = b"".join(content)
        if name in ready:  # git sends no content when it asks again for a delayed blob
          data = pointers.pop(name)
        try:
          result = ready.pop(name).result() if name in ready else smudge(data, os.fsdecode(name), exclude,
                                                                       pool if fields.get(b"can-delay") == b"1" else None)
        except Exception as error:
          print(f"lfs.py: {os.fsdecode(name)}: {error}; leaving the pointer, run ./lfs.py pull to retry", file=sys.stderr)
          result = data
      if isinstance(result, concurrent.futures.Future):
        pending[name] = result
        pointers[name] = data
        write_packets(b"status=delayed\n")
        continue
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


def prune():
  heads, blobs = [], set()
  for directory in [GIT_DIR, *(GIT_DIR / "worktrees").glob("*")]:
    args = ("-C", str(ROOT), "--git-dir", str(directory))
    if head := git(*args, "rev-parse", "--verify", "--quiet", "HEAD", check=False).decode().strip():
      heads.append(head)
    entries = git(*args, "ls-files", "--stage", "-z")
    blobs.update(entry.split(b"\t", 1)[0].split()[1].decode() for entry in entries.split(b"\0") if entry)
  stashes = git("reflog", "show", "--format=%H %P", "refs/stash", check=False).decode().split()
  for ref in set(heads + stashes):
    entries = git("ls-tree", "-r", "-z", ref)
    blobs.update(entry.split(b"\t", 1)[0].split()[2].decode() for entry in entries.split(b"\0") if entry)
  remote = git("config", "--get", "--default", "origin", "lfs.pruneremotetocheck").decode().strip()
  blobs.update(git("rev-list", "--objects", "--no-object-names", *heads, *stashes, "--all", "--not",
                   f"--remotes={remote}").decode().splitlines())
  retained = {parsed[0] for _oid, data in cat_small(blobs) if (parsed := parse_pointer(data))}
  cutoff = time.time() - 7 * 24 * 60 * 60  # noqa: TID251
  removed, freed = 0, 0
  for path in STORE.glob("??/??/*"):
    if path.name in retained or not re.fullmatch(r"[0-9a-f]{64}", path.name):
      continue
    with object_lock(path.name):
      if path.is_file() and path.stat().st_mtime < cutoff:
        freed += path.stat().st_size
        path.unlink()
        removed += 1
  print(f"Pruned {removed} objects ({freed / 1024**2:.1f} MiB)")


def pull(exclude=None):
  install()
  found = pointers_for_index()
  if exclude is None:
    exclude = git("config", "--get", "lfs.fetchexclude", check=False).decode().strip()
  if exclude:
    found = {obj: kept for obj, paths in found.items() if (kept := [path for path in paths if not excluded(path, exclude)])}
  parallel(download_one, found)
  materialized = []
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
      materialized.append(name)
  if materialized:
    # Clear pointer stat data without changing staged blob IDs or modes.
    entries = git("--literal-pathspecs", "-C", str(ROOT), "ls-files", "--stage", "-z", "--", *materialized)
    git("update-index", "-z", "--index-info", stdin=entries)
    git("update-index", "-q", "--refresh", check=False)


def bucket_request(path, data=None):
  headers = {"Content-Type": "application/x-ndjson"} if data is not None else {}
  request = hf_request(f"https://huggingface.co/api/buckets/{BUCKET}/{path}", data=data, headers=headers)
  for attempt in range(4):
    try:
      with urllib.request.urlopen(request, timeout=30) as response:
        return response.read(), response.headers
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError) as error:
      retryable = not isinstance(error, urllib.error.HTTPError) or error.code in (408, 429) or 500 <= error.code < 600
      if not retryable or attempt == 3:
        raise
      time.sleep(2 ** attempt)


def upload_xet(objects, upload_files):
  def connection():
    _, headers = bucket_request("xet-write-token")
    return headers["X-Xet-Cas-Url"], (headers["X-Xet-Access-Token"], int(headers["X-Xet-Token-Expiration"]))

  oids = list(objects)
  for start in range(0, len(oids), 100):
    batch = oids[start:start + 100]
    paths = [object_path(oid) for oid in batch]
    endpoint, token = connection()
    infos = upload_files([str(path) for path in paths], endpoint, token, lambda: connection()[1], None, "bucket", skip_sha256=True)
    payload = b"".join(json.dumps({"type": "addFile", "path": f"sha256/{oid}", "xetHash": info.hash,
                                  "mtime": int(path.stat().st_mtime * 1000)}).encode() + b"\n"
                       for oid, path, info in zip(batch, paths, infos, strict=True))
    bucket_request("batch", payload)


def upload(objects):
  objects = dict(objects)
  if not objects:
    return
  existing = object_info(objects)
  pending = {}
  for oid, size in objects.items():
    key = f"sha256/{oid}"
    if key in existing:
      if existing[key] != size:
        raise RuntimeError(f"invalid bucket object {oid}: size {existing[key]}, expected {size}")
      continue
    pending[oid] = size
  if not pending:
    return
  try:
    from hf_xet import upload_files
  except ImportError:
    raise RuntimeError("Missing hf-xet: run 'uv sync' and activate .venv.") from None
  for oid, size in pending.items():
    path = object_path(oid)
    if not valid(path, size):
      raise RuntimeError(f"missing local LFS object {oid}")
    if hash_file(path) != oid:
      raise RuntimeError(f"local sha256 mismatch for {oid}")
  upload_xet(pending, upload_files)
  uploaded = object_info(pending)
  for oid, size in pending.items():
    if uploaded.get(f"sha256/{oid}") != size:
      raise RuntimeError(f"uploaded object {oid} is missing or has the wrong size")


def pre_push(remote):
  if os.environ.get("GIT_LFS_SKIP_PUSH", "").upper() in TRUE_VALUES:
    return
  input_data = sys.stdin.buffer.read()
  updates = (line.decode().split() for line in input_data.splitlines())
  local_oids = [local_oid for _local_ref, local_oid, _remote_ref, _remote_oid in updates if local_oid != ZERO]
  if not local_oids:
    return
  output = git("rev-list", "--objects", "--no-object-names", *local_oids, "--not", f"--remotes={remote}")
  object_ids = output.decode().splitlines()
  objects = [parsed for _oid, data in cat_small(object_ids) if (parsed := parse_pointer(data))]
  try:
    upload(objects)
  except urllib.error.HTTPError as error:
    if error.code not in (401, 403):
      raise
    sys.exit(f"lfs.py: uploading LFS objects to {BUCKET} needs a Hugging Face token with write access (hf auth login, or set HF_TOKEN)")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", choices=("filter-process", "pull", "pre-push", "install", "ls-files", "prune"))
  parser.add_argument("remote", nargs="?", default="origin")
  parser.add_argument("url", nargs="?")
  parser.add_argument("--exclude", help="comma-separated path globs to leave as pointers; overrides lfs.fetchexclude")
  args = parser.parse_args()
  if args.command == "filter-process":
    filter_process()
  elif args.command == "pull":
    pull(args.exclude)
  elif args.command == "pre-push":
    pre_push(args.remote)
  elif args.command == "install":
    install()
  elif args.command == "prune":
    prune()
  elif args.command == "ls-files":
    for (oid, _size), paths in pointers_for_index().items():
      for path in paths:
        print(f"{oid} {path}")


if __name__ == "__main__":
  main()
