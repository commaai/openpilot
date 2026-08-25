#!/usr/bin/env python3
"""A small, curl-backed Git LFS client."""

import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

POINTER_VERSION = "https://git-lfs.github.com/spec/v1"
POINTER_RE = re.compile(rb"\Aversion " + POINTER_VERSION.encode() + rb"\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n\Z")
ENDPOINT = "https://gitlab.com/commaai/openpilot-lfs.git/info/lfs"
ZERO = "0" * 40
TRANSFERS = int(os.environ.get("GIT_XLFS_TRANSFERS", "4"))
RANGES = int(os.environ.get("GIT_XLFS_RANGES", "4"))
RANGE_MIN = int(os.environ.get("GIT_XLFS_RANGE_MIN", str(16 << 20)))


def git(*args, stdin=None, check=True):
  return subprocess.run(["git", *args], input=stdin, capture_output=True, check=check).stdout


ROOT = Path(git("rev-parse", "--show-toplevel").decode().strip())
GIT_DIR = Path(git("rev-parse", "--git-dir").decode().strip())
if not GIT_DIR.is_absolute():
  GIT_DIR = ROOT / GIT_DIR
STORE = GIT_DIR / "lfs" / "objects"


def install():
  for key, value in {
    "filter.lfs.process": "./lfs.py filter-process",
    "filter.lfs.clean": "./lfs.py clean -- %f",
    "filter.lfs.smudge": "./lfs.py smudge -- %f",
    "filter.lfs.required": "true",
  }.items():
    git("config", "--local", key, value)
  hook = Path(git("rev-parse", "--git-path", "hooks").decode().strip()) / "pre-push"
  if not hook.is_absolute():
    hook = ROOT / hook
  hook.parent.mkdir(parents=True, exist_ok=True)
  hook.write_text('#!/bin/sh\nexec "$(git rev-parse --show-toplevel)/lfs.py" pre-push "$@"\n')
  hook.chmod(0o755)


def object_path(oid):
  return STORE / oid[:2] / oid[2:4] / oid


def lock_path(oid):
  return GIT_DIR / "lfs" / "locks" / (oid + ".lock")


def pointer(oid, size):
  return f"version {POINTER_VERSION}\noid sha256:{oid}\nsize {size}\n".encode()


def parse_pointer(data):
  match = POINTER_RE.match(data)
  return (match[1].decode(), int(match[2])) if match else None


class Lock:
  def __init__(self, path):
    self.path = path

  def __enter__(self):
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.file = self.path.open("a+b")
    fcntl.flock(self.file, fcntl.LOCK_EX)

  def __exit__(self, *_):
    self.file.close()


class Reauth(Exception):
  pass


def valid(path, oid, size):
  return path.is_file() and path.stat().st_size == size


def hash_file(path):
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1 << 20):
      digest.update(chunk)
  return digest.hexdigest()


def store(chunks):
  STORE.mkdir(parents=True, exist_ok=True)
  fd, name = tempfile.mkstemp(prefix="incoming-", suffix=".tmp", dir=STORE)
  digest, size = hashlib.sha256(), 0
  try:
    with os.fdopen(fd, "wb") as out:
      for chunk in chunks:
        out.write(chunk)
        digest.update(chunk)
        size += len(chunk)
    tmp = Path(name)
    if size <= 1024:
      data = tmp.read_bytes()
      if parse_pointer(data):
        tmp.unlink()
        return data
    oid = digest.hexdigest()
    dst = object_path(oid)
    with Lock(lock_path(oid)):
      if not valid(dst, oid, size):
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, dst)
      else:
        tmp.unlink()
    return pointer(oid, size)
  except BaseException:
    Path(name).unlink(missing_ok=True)
    raise


class Client:
  def __init__(self, operation):
    self.operation = operation
    self.auth = {}

  def curl(self, method, url, headers=(), data=None, output=None, byte_range=None, upload=None):
    args = [
      "curl",
      "--silent",
      "--show-error",
      "--location",
      "--retry",
      "3",
      "--retry-all-errors",
      "--connect-timeout",
      "20",
      "-X",
      method,
      "--write-out",
      "%{http_code}",
    ]
    if output is not None:
      args += ["-o", str(output)]
    for key, value in {**self.auth, **dict(headers)}.items():
      args += ["-H", f"{key}: {value}"]
    if byte_range:
      args += ["--range", byte_range]
    if data is not None:
      args += ["--data-binary", "@-"]
    if upload is not None:
      args += ["--upload-file", str(upload)]
    raw = subprocess.run(args + [url], input=data, stdout=subprocess.PIPE).stdout
    return int(raw[-3:] or b"0"), raw[:-3] if output is None else b""

  def ssh_auth(self):
    result = subprocess.run(["ssh", "git@gitlab.com", "git-lfs-authenticate", "commaai/openpilot-lfs.git", self.operation], stdout=subprocess.PIPE, check=True)
    reply = json.loads(result.stdout)
    self.auth = reply.get("header", {})

  def request(self, method, url, headers=(), data=None):
    status, body = self.curl(method, url, headers, data)
    if status == 401 and self.operation == "upload":
      self.ssh_auth()
      status, body = self.curl(method, url, headers, data)
    if not 200 <= status < 300:
      raise RuntimeError(f"HTTP {status} from {url}: {body[:500].decode(errors='replace')}")
    return body

  def reauth(self):
    if self.operation != "upload":
      raise RuntimeError("public LFS download authorization failed")
    self.ssh_auth()

  def batch(self, objects):
    body = json.dumps({"operation": self.operation, "transfers": ["basic"], "objects": [{"oid": oid, "size": size} for oid, size in objects]}).encode()
    headers = {"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"}
    reply = json.loads(self.request("POST", ENDPOINT + "/objects/batch", headers, body))
    if reply.get("transfer", "basic") != "basic":
      raise RuntimeError(f"unsupported transfer adapter: {reply['transfer']}")
    return reply.get("objects", [])


def action_headers(action):
  return action.get("header", {}).items()


def download_one(client, oid, size, action):
  dst = object_path(oid)
  with Lock(lock_path(oid)):
    if valid(dst, oid, size):
      return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{oid}-{os.getpid()}.tmp")
    href = action["href"]
    try:
      ranges = min(RANGES, max(1, (size + RANGE_MIN - 1) // RANGE_MIN))
      if ranges > 1:
        spans = [(i * size // ranges, (i + 1) * size // ranges - 1) for i in range(ranges)]
        parts = [dst.with_name(f"{oid}-{os.getpid()}-{i}.tmp") for i in range(ranges)]

        def fetch(item):
          part, span = item
          return client.curl("GET", href, action_headers(action), output=part, byte_range=f"{span[0]}-{span[1]}")[0]

        with concurrent.futures.ThreadPoolExecutor(max_workers=ranges) as pool:
          statuses = list(pool.map(fetch, zip(parts, spans, strict=True)))
        if 401 in statuses:
          raise Reauth
        if all(status == 206 and part.stat().st_size == end - start + 1 for status, part, (start, end) in zip(statuses, parts, spans, strict=True)):
          with tmp.open("wb") as out:
            for part in parts:
              with part.open("rb") as stream:
                shutil.copyfileobj(stream, out)
        else:
          for part in parts:
            part.unlink(missing_ok=True)
          status, _ = client.curl("GET", href, action_headers(action), output=tmp)
          if status == 401:
            raise Reauth
          if not 200 <= status < 300:
            raise RuntimeError(f"download returned HTTP {status}")
        for part in parts:
          part.unlink(missing_ok=True)
      else:
        status, _ = client.curl("GET", href, action_headers(action), output=tmp)
        if status == 401:
          raise Reauth
        if not 200 <= status < 300:
          raise RuntimeError(f"download returned HTTP {status}")
      if tmp.stat().st_size != size or hash_file(tmp) != oid:
        raise RuntimeError(f"sha256 mismatch for {oid}")
      os.replace(tmp, dst)
      return dst
    finally:
      tmp.unlink(missing_ok=True)


def download_many(objects):
  missing = [(oid, size) for oid, size in objects if not valid(object_path(oid), oid, size)]
  if not missing:
    return
  client = Client("download")
  for attempt in range(2):
    actions = {}
    for start in range(0, len(missing), 100):
      for obj in client.batch(missing[start : start + 100]):
        if obj.get("error"):
          raise RuntimeError(f"{obj['oid']}: {obj['error'].get('message', 'server error')}")
        if "download" not in obj.get("actions", {}):
          raise RuntimeError(f"server gave no download action for {obj['oid']}")
        actions[obj["oid"]] = obj["actions"]["download"]
    try:
      with concurrent.futures.ThreadPoolExecutor(max_workers=TRANSFERS) as pool:
        for future in [pool.submit(download_one, client, oid, size, actions[oid]) for oid, size in missing]:
          future.result()
      return
    except Reauth:
      if attempt:
        raise
      client.reauth()


def smudge(data):
  if os.environ.get("GIT_LFS_SKIP_SMUDGE") == "1":
    return data
  parsed = parse_pointer(data)
  if not parsed:
    return data
  oid, size = parsed
  path = object_path(oid)
  if not valid(path, oid, size):
    try:
      download_many([parsed])
    except Exception as error:
      print(f"git-xlfs: warning: could not download {oid}: {error}", file=sys.stderr)
      return data
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


def write_packet(data):
  sys.stdout.buffer.write(f"{len(data) + 4:04x}".encode() + data)


def write_packets(source):
  if isinstance(source, bytes):
    source = (source,)
  for chunk in source:
    for start in range(0, len(chunk), 65512):
      write_packet(chunk[start : start + 65512])
  sys.stdout.buffer.write(b"0000")


def filter_process():
  hello = b"".join(read_packets(sys.stdin.buffer))
  if b"git-filter-client\n" not in hello or b"version=2\n" not in hello:
    raise RuntimeError("expected Git filter protocol v2")
  write_packet(b"git-filter-server\n")
  write_packet(b"version=2\n")
  sys.stdout.buffer.write(b"0000")
  sys.stdout.buffer.flush()
  capabilities = b"".join(read_packets(sys.stdin.buffer))
  if b"capability=clean\n" not in capabilities or b"capability=smudge\n" not in capabilities:
    raise RuntimeError("Git did not offer clean and smudge capabilities")
  write_packet(b"capability=clean\n")
  write_packet(b"capability=smudge\n")
  sys.stdout.buffer.write(b"0000")
  sys.stdout.buffer.flush()
  while True:
    try:
      fields = dict(line.rstrip(b"\n").split(b"=", 1) for line in read_packets(sys.stdin.buffer))
    except EOFError:
      return
    command = fields.get(b"command", b"").decode()
    if command == "clean":
      result = store(read_packets(sys.stdin.buffer))
    elif command == "smudge":
      result = smudge(b"".join(read_packets(sys.stdin.buffer)))
    else:
      list(read_packets(sys.stdin.buffer))
      result = b""
    write_packet(b"status=success\n")
    sys.stdout.buffer.write(b"0000")
    if isinstance(result, Path):
      with result.open("rb") as source_stream:
        write_packets(iter(lambda: source_stream.read(65512), b""))
    else:
      write_packets(result)
    sys.stdout.buffer.write(b"0000")
    sys.stdout.buffer.flush()


def cat_small(oids):
  unique = list(dict.fromkeys(oids))
  checked = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)", stdin=("\n".join(unique) + "\n").encode()).decode().splitlines()
  small = [line.split()[0] for line in checked if len(line.split()) == 3 and line.split()[1] == "blob" and int(line.split()[2]) <= 1024]
  if not small:
    return {}
  proc = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
  out = {}
  for oid in small:
    proc.stdin.write((oid + "\n").encode())
    proc.stdin.flush()
    size = int(proc.stdout.readline().split()[2])
    out[oid] = proc.stdout.read(size)
    proc.stdout.read(1)
  proc.stdin.close()
  proc.wait()
  return out


def pointers_for_ref(ref):
  found = {}
  matches = git("grep", "-l", "-z", "-e", f"^version {POINTER_VERSION}$", ref, "--").split(b"\0")
  for match in matches:
    if not match:
      continue
    path = match.split(b":", 1)[1].decode(errors="surrogateescape")
    parsed = parse_pointer(git("show", f"{ref}:{path}"))
    if parsed:
      found.setdefault(parsed, []).append(path)
  return found


def pull(ref="HEAD", exclude=None):
  found = pointers_for_ref(ref)
  if exclude is None:
    exclude = git("config", "--get", "lfs.fetchexclude", check=False).decode().strip()
  excluded = {path.strip() for path in exclude.split(",") if path.strip()}
  if excluded:
    found = {obj: kept for obj, paths in found.items() if (kept := [path for path in paths if path not in excluded])}
  download_many(found)
  if git("rev-parse", ref).strip() == git("rev-parse", "HEAD").strip():
    materialized = []
    for (oid, size), paths in found.items():
      for name in paths:
        path = ROOT / name
        try:
          if path.stat().st_size > 1024 or parse_pointer(path.read_bytes()) != (oid, size):
            continue
        except (FileNotFoundError, IsADirectoryError):
          continue
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + "-", suffix=".tmp", dir=path.parent)
        with os.fdopen(fd, "wb") as out, object_path(oid).open("rb") as source:
          shutil.copyfileobj(source, out)
        os.chmod(tmp_name, path.stat().st_mode)
        os.replace(tmp_name, path)
        materialized.append(name)
    if materialized:
      git("add", "--renormalize", "--", *materialized)


def upload_one(client, oid, size, actions):
  path = object_path(oid)
  if not valid(path, oid, size):
    raise RuntimeError(f"missing local LFS object {oid}")
  upload = actions.get("upload")
  if not upload:
    return
  status, _ = client.curl("PUT", upload["href"], action_headers(upload), output=Path(os.devnull), upload=path)
  if status == 401:
    raise Reauth
  if not 200 <= status < 300:
    raise RuntimeError(f"upload of {oid} returned HTTP {status}")
  if verify := actions.get("verify"):
    body = json.dumps({"oid": oid, "size": size}).encode()
    headers = dict(action_headers(verify)) | {"Content-Type": "application/vnd.git-lfs+json"}
    client.request("POST", verify["href"], headers, body)


def pre_push(remote):
  if os.environ.get("GIT_LFS_SKIP_PUSH") == "1":
    return
  lines = sys.stdin.buffer.read().decode().splitlines()
  ranges = []
  for line in lines:
    _local_ref, local_oid, _remote_ref, remote_oid = line.split()
    if local_oid.startswith(ZERO):
      continue
    have_remote = (
      not remote_oid.startswith(ZERO) and subprocess.run(["git", "cat-file", "-e", remote_oid + "^{commit}"], stderr=subprocess.DEVNULL).returncode == 0
    )
    ranges.append(f"{remote_oid}..{local_oid}" if have_remote else local_oid)
  if not ranges:
    return
  args = ["rev-list", "--objects", *ranges]
  if any(".." not in item for item in ranges):
    args += ["--not", f"--remotes={remote}"]
  object_ids = [line.split(b" ", 1)[0].decode() for line in git(*args).splitlines()]
  contents = cat_small(object_ids)
  objects = list(dict.fromkeys(parsed for data in contents.values() if (parsed := parse_pointer(data))))
  client = Client("upload")
  for start in range(0, len(objects), 100):
    batch = objects[start : start + 100]
    for attempt in range(2):
      replies = client.batch(batch)
      try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=TRANSFERS) as pool:
          jobs = []
          for obj in replies:
            if obj.get("error"):
              raise RuntimeError(f"{obj['oid']}: {obj['error'].get('message', 'server error')}")
            jobs.append(pool.submit(upload_one, client, obj["oid"], obj["size"], obj.get("actions", {})))
          for job in jobs:
            job.result()
        break
      except Reauth:
        if attempt:
          raise
        client.reauth()


def main():
  command = sys.argv[1] if len(sys.argv) > 1 else ""
  if command == "filter-process":
    filter_process()
  elif command == "clean":
    sys.stdout.buffer.write(store(iter(lambda: sys.stdin.buffer.read(1 << 20), b"")))
  elif command == "smudge":
    result = smudge(sys.stdin.buffer.read())
    if isinstance(result, Path):
      with result.open("rb") as stream:
        shutil.copyfileobj(stream, sys.stdout.buffer)
    else:
      sys.stdout.buffer.write(result)
  elif command == "pull":
    args = sys.argv[2:]
    exclude = next((arg.split("=", 1)[1] for arg in args if arg.startswith("--exclude=")), None)
    pull(next((arg for arg in args if not arg.startswith("--")), "HEAD"), exclude)
  elif command == "pre-push":
    pre_push(sys.argv[2] if len(sys.argv) > 2 else "origin")
  elif command == "install":
    install()
  else:
    raise SystemExit("usage: lfs.py {filter-process|clean|smudge|pull|pre-push|install}")


if __name__ == "__main__":
  main()
