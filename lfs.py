#!/usr/bin/env python3
"""A small, curl-backed Git LFS client."""

import concurrent.futures
import contextlib
import fcntl
import hashlib
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
import xml.etree.ElementTree as ET

POINTER_VERSION = "https://git-lfs.github.com/spec/v1"
POINTER_RE = re.compile(rb"\Aversion " + POINTER_VERSION.encode() + rb"\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n\Z")
# Temporary development bucket. Production cutover requires explicit approval.
R2_BASE = "https://4a5b3dbb5151b29bca371371370cbc54.r2.cloudflarestorage.com/fakelfs-test/sha256"
PUBLIC_BASE = "https://pub-8fc0b2b05a7e46349231e9c61e182006.r2.dev/sha256"
ZERO = "0" * 40
TRANSFERS = 16
RANGES = 8
RANGE_MIN = 16 << 20
PART_SIZE = 128 << 20
UPLOAD_CONCURRENCY = 8
RETRIES = 5
CURL_OPTIONS = ("--silent", "--connect-timeout", "20", "--speed-limit", "1024", "--speed-time", "30")
RETRY_OPTIONS = ("--retry", str(RETRIES), "--retry-all-errors", "--remove-on-error")
R2_OPTIONS = (
  "--variable",
  "%R2_ACCESS_KEY_ID",
  "--variable",
  "%R2_SECRET_ACCESS_KEY",
  "--expand-user",
  "{{R2_ACCESS_KEY_ID}}:{{R2_SECRET_ACCESS_KEY}}",
  "--aws-sigv4",
  "aws:amz:auto:s3",
)


def git(*args, stdin=None, check=True):
  return subprocess.run(["git", *args], input=stdin, capture_output=True, check=check).stdout


ROOT, GIT_DIR = map(Path, git("rev-parse", "--show-toplevel", "--absolute-git-dir").decode().splitlines())
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
    has_lfs_guard = len(lines) == 3 and lines[1].startswith("command -v git-lfs ")
    if lines[:1] == ["#!/bin/sh"] and lines[-1] in commands and (len(lines) == 2 or has_lfs_guard):
      path.unlink()


def install():
  filter_process_command = shlex.join((sys.executable, str(Path(__file__).resolve()), "filter-process"))
  legacy_hook = ROOT / Path(git("rev-parse", "--git-path", "hooks/pre-push").decode().strip())
  remove_git_lfs_hooks(legacy_hook.parent)
  hook = GIT_DIR / "lfs-hooks" / "pre-push"
  if legacy_hook.parent != hook.parent:
    shutil.copytree(legacy_hook.parent, hook.parent, dirs_exist_ok=True, symlinks=True)
  hook_script = '#!/bin/sh\nexec "$(git rev-parse --show-toplevel)/lfs.py" pre-push "$@"\n'
  if hook.is_file() and hook.read_text(errors="replace") != hook_script:
    raise RuntimeError(f"refusing to replace custom hook {hook}")
  hook.parent.mkdir(parents=True, exist_ok=True)
  hook.write_text(hook_script)
  hook.chmod(0o755)
  git("config", "--local", "extensions.worktreeConfig", "true")
  git("config", "--local", "--remove-section", "filter.lfs", check=False)
  git("config", "--worktree", "filter.lfs.process", filter_process_command)
  git("config", "--worktree", "filter.lfs.required", "true")
  git("config", "--worktree", "core.hooksPath", str(hook.parent))


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
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


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


def curl(url, *options, data=None, output=None):
  marker = b"\n\x1e"
  write_out = marker.decode() + "%{http_code}\n%header{etag}\n%header{content-length}"
  args = ["curl", "--disable", *CURL_OPTIONS, *options, "--write-out", write_out]
  if output is not None:
    args += ["--output", str(output)]
  args.append(url)
  result = subprocess.run(args, input=data, capture_output=True)
  if result.returncode or marker not in result.stdout:
    return 0, "", "", b""
  body, metadata = result.stdout.rsplit(marker, 1)
  status, etag, content_length = metadata.decode().split("\n")
  return int(status), etag, content_length, body if output is None else b""


def public_object_exists(oid, size):
  status, _etag, content_length, _body = curl(f"{PUBLIC_BASE}/{oid}", *RETRY_OPTIONS, "--head")
  if status == 404:
    return False
  if status != 200 or not content_length.isdecimal() or int(content_length) != size:
    raise RuntimeError(f"invalid public object {oid}: HTTP {status}, size {content_length!r}")
  return True


def parallel(function, arguments, workers=TRANSFERS):
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    futures = [pool.submit(function, *args) for args in arguments]
    return [future.result() for future in futures]


def download_one(oid, size):
  dst = object_path(oid)
  with object_lock(oid):
    if valid(dst, size):
      return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    href = f"{PUBLIC_BASE}/{oid}"
    with tempfile.TemporaryDirectory(prefix=f"{oid}-", dir=dst.parent) as directory:
      tmp = Path(directory) / "object"
      ranges = min(RANGES, max(1, (size + RANGE_MIN - 1) // RANGE_MIN))
      if ranges > 1:
        spans = [(i * size // ranges, (i + 1) * size // ranges - 1) for i in range(ranges)]
        parts = [Path(directory) / str(i) for i in range(ranges)]

        def fetch(part, span):
          curl(href, *RETRY_OPTIONS, "--range", f"{span[0]}-{span[1]}", output=part)

        parallel(fetch, zip(parts, spans, strict=True), ranges)
        with tmp.open("wb") as out:
          for part in parts:
            with part.open("rb") as stream:
              shutil.copyfileobj(stream, out)
      else:
        curl(href, *RETRY_OPTIONS, output=tmp)
      if tmp.stat().st_size != size or hash_file(tmp) != oid:
        raise RuntimeError(f"sha256 mismatch for {oid}")
      os.replace(tmp, dst)
      return dst


def smudge(data):
  if os.environ.get("GIT_LFS_SKIP_SMUDGE") == "1":
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
    result = store(content) if fields[b"command"] == b"clean" else smudge(b"".join(content))
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


def pointers_for_head():
  paths = {}
  for entry in filter(None, git("ls-tree", "-rlz", "HEAD").split(b"\0")):
    metadata, path = entry.split(b"\t", 1)
    _mode, kind, oid, size = metadata.split()
    if kind == b"blob" and int(size) <= 1024:
      paths.setdefault(oid.decode(), []).append(path.decode(errors="surrogateescape"))
  found = {}
  for oid, data in cat_small(paths):
    if parsed := parse_pointer(data):
      found.setdefault(parsed, []).extend(paths[oid])
  return found


def pull():
  install()
  found = pointers_for_head()
  exclude = git("config", "--get", "lfs.fetchexclude", check=False).decode().strip()
  if exclude:
    found = {obj: kept for obj, paths in found.items() if (kept := [path for path in paths if path != exclude])}
  parallel(download_one, found)
  materialized = []
  for (oid, size), paths in found.items():
    for name in paths:
      path = ROOT / name
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
    git("add", "--renormalize", "--", *materialized)


class ObjectExists(Exception):
  pass


def r2_request(action, oid, *options, query=(), data=None, statuses=(200,), race_size=None):
  if not os.environ.get("R2_ACCESS_KEY_ID") or not os.environ.get("R2_SECRET_ACCESS_KEY"):
    raise RuntimeError("R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY are required for uploads")
  request_options = list(R2_OPTIONS) + list(options)
  for key, value in sorted(query):
    request_options += ["--url-query", f"{key}={value}"]
  for attempt in range(RETRIES):
    status, etag, _content_length, body = curl(f"{R2_BASE}/{oid}", *request_options, data=data)
    if status in statuses:
      return etag, body
    if race_size is not None and public_object_exists(oid, race_size):
      raise ObjectExists
    if status not in (0, 429, 500, 502, 503, 504):
      break
    time.sleep(min(2**attempt, 8))
  raise RuntimeError(f"{action} returned HTTP {status} for {oid}")


def xml_value(body, name):
  try:
    element = ET.fromstring(body).find(f".//{{*}}{name}")
  except ET.ParseError as error:
    raise RuntimeError("R2 returned malformed XML") from error
  if element is None or element.text is None:
    raise RuntimeError(f"R2 response has no {name}")
  return element.text


def prepare_part(path, offset, size, part_number, directory):
  part_path = Path(directory) / str(part_number)
  md5 = hashlib.md5(usedforsecurity=False)
  with path.open("rb") as source, part_path.open("xb") as part:
    source.seek(offset)
    remaining = size
    while remaining:
      chunk = source.read(min(1 << 20, remaining))
      if not chunk:
        raise RuntimeError(f"short read from {path}")
      part.write(chunk)
      md5.update(chunk)
      remaining -= len(chunk)
  return part_path, md5.hexdigest()


def create_multipart_upload(oid, size):
  _etag, body = r2_request("CreateMultipartUpload", oid, "--data-binary", "@-", query=(("uploads", ""),), data=b"", race_size=size)
  return xml_value(body, "UploadId")


def upload_part(path, oid, size, upload_id, part_number, offset, part_size, directory):
  part_path, part_md5 = prepare_part(path, offset, part_size, part_number, directory)
  query = (("partNumber", part_number), ("uploadId", upload_id))
  etag, _body = r2_request(f"UploadPart {part_number}", oid, "--upload-file", str(part_path), query=query, race_size=size)
  etag = etag.strip('"').lower()
  if etag != part_md5:
    raise RuntimeError(f"wrong ETag for part {part_number} of {oid}")
  part_path.unlink()
  return part_number, etag


def abort_multipart_upload(oid, upload_id):
  r2_request("AbortMultipartUpload", oid, "--request", "DELETE", query=(("uploadId", upload_id),), statuses=(204, 404))


def complete_multipart_upload(oid, size, upload_id, parts):
  root = ET.Element("CompleteMultipartUpload")
  for part_number, etag in sorted(parts):
    part = ET.SubElement(root, "Part")
    ET.SubElement(part, "PartNumber").text = str(part_number)
    ET.SubElement(part, "ETag").text = f'"{etag}"'
  body = ET.tostring(root, encoding="utf-8", xml_declaration=True)
  _etag, response = r2_request("CompleteMultipartUpload", oid, "--data-binary", "@-", query=(("uploadId", upload_id),), data=body, race_size=size)
  xml_value(response, "ETag")
  if not public_object_exists(oid, size):
    raise RuntimeError(f"completed object {oid} is not publicly readable")


def multipart_upload(path, oid, size):
  upload_id = create_multipart_upload(oid, size)
  parts = [(part_number + 1, offset, min(PART_SIZE, size - offset)) for part_number, offset in enumerate(range(0, size, PART_SIZE))]
  try:
    with tempfile.TemporaryDirectory(prefix="r2-upload-", dir=STORE) as directory:
      arguments = [(path, oid, size, upload_id, *part, directory) for part in parts]
      uploaded_parts = parallel(upload_part, arguments, UPLOAD_CONCURRENCY)
      complete_multipart_upload(oid, size, upload_id, uploaded_parts)
  except BaseException as error:
    try:
      abort_multipart_upload(oid, upload_id)
    except Exception as abort_error:
      error.add_note(str(abort_error))
    raise


def upload_one(oid, size):
  path = object_path(oid)
  if public_object_exists(oid, size):
    return
  if not valid(path, size):
    raise RuntimeError(f"missing local LFS object {oid}")
  if hash_file(path) != oid:
    raise RuntimeError(f"local sha256 mismatch for {oid}")
  with contextlib.suppress(ObjectExists):
    multipart_upload(path, oid, size)


def pre_push(remote):
  if os.environ.get("GIT_LFS_SKIP_PUSH") == "1":
    return
  updates = (line.decode().split() for line in sys.stdin.buffer)
  local_oids = [local_oid for _local_ref, local_oid, _remote_ref, _remote_oid in updates if local_oid != ZERO]
  if not local_oids:
    return
  output = git("rev-list", "--objects", "--no-object-names", *local_oids, "--not", f"--remotes={remote}")
  object_ids = output.decode().splitlines()
  parallel(upload_one, (parsed for _oid, data in cat_small(object_ids) if (parsed := parse_pointer(data))))


def main():
  command = sys.argv[1] if len(sys.argv) > 1 else ""
  if command == "filter-process":
    filter_process()
  elif command == "pull":
    pull()
  elif command == "pre-push":
    pre_push(sys.argv[2] if len(sys.argv) > 2 else "origin")
  elif command == "install":
    install()
  else:
    raise SystemExit("usage: lfs.py {filter-process|pull|pre-push|install}")


if __name__ == "__main__":
  main()
