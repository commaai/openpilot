#!/usr/bin/env python3
"""A small, curl-backed Git LFS client."""

import concurrent.futures
import fcntl
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import xml.etree.ElementTree as ET

POINTER_VERSION = "https://git-lfs.github.com/spec/v1"
POINTER_RE = re.compile(rb"\Aversion " + POINTER_VERSION.encode() + rb"\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n\Z")
# Temporary development bucket. Production cutover requires explicit approval.
BUCKET = "fakelfs-test"
S3_ENDPOINT = "https://4a5b3dbb5151b29bca371371370cbc54.r2.cloudflarestorage.com"
PUBLIC_BASE = "https://pub-8fc0b2b05a7e46349231e9c61e182006.r2.dev"
ZERO = "0" * 40
TRANSFERS = 16
RANGES = 8
RANGE_MIN = 16 << 20
MULTIPART_THRESHOLD = 100 << 20
PART_SIZE = 128 << 20
UPLOAD_CONCURRENCY = 8
RETRIES = 5
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def git(*args, stdin=None, check=True):
  return subprocess.run(["git", *args], input=stdin, capture_output=True, check=check).stdout


ROOT = Path(git("rev-parse", "--show-toplevel").decode().strip())
GIT_DIR = Path(git("rev-parse", "--git-dir").decode().strip())
if not GIT_DIR.is_absolute():
  GIT_DIR = ROOT / GIT_DIR
STORE = GIT_DIR / "lfs" / "objects"


def install():
  filter_process_command = shlex.join((sys.executable, str(Path(__file__).resolve()), "filter-process"))
  for key, value in {
    "filter.lfs.process": filter_process_command,
    "filter.lfs.required": "true",
  }.items():
    git("config", "--local", key, value)
  git("config", "--local", "--unset-all", "filter.lfs.clean", check=False)
  git("config", "--local", "--unset-all", "filter.lfs.smudge", check=False)
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


def valid(path, oid, size):
  return path.is_file() and path.stat().st_size == size


def hash_file(path):
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1 << 20):
      digest.update(chunk)
  return digest.hexdigest()


def hash_region(path, offset, size):
  sha256 = hashlib.sha256()
  md5 = hashlib.md5(usedforsecurity=False)
  with path.open("rb") as stream:
    stream.seek(offset)
    remaining = size
    while remaining:
      chunk = stream.read(min(1 << 20, remaining))
      if not chunk:
        raise RuntimeError(f"short read from {path}")
      sha256.update(chunk)
      md5.update(chunk)
      remaining -= len(chunk)
  return sha256.hexdigest(), md5.hexdigest()


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


def curl_config(headers, credentials=None):
  values = list(headers.values())
  if credentials:
    values.extend(credentials)
  for value in values:
    if any(character in value for character in "\\\r\n\""):
      raise RuntimeError("invalid curl configuration value")
  fd, name = tempfile.mkstemp(prefix="r2-headers-", suffix=".conf")
  os.chmod(name, 0o600)
  with os.fdopen(fd, "w") as config:
    for key, value in headers.items():
      config.write(f'header = "{key}: {value}"\n')
    if credentials:
      access_key, secret_key = credentials
      config.write(f'user = "{access_key}:{secret_key}"\n')
      config.write('aws-sigv4 = "aws:amz:auto:s3"\n')
  return Path(name)


def parse_headers(path):
  headers = {}
  for line in path.read_text(errors="replace").splitlines():
    if line.startswith("HTTP/"):
      headers = {}
    elif ":" in line:
      key, value = line.split(":", 1)
      headers[key.lower()] = value.strip()
  return headers


def curl(method, url, *, headers=None, credentials=None, data=None, output=None, byte_range=None, upload=None, retry=False):
  header_fd, header_name = tempfile.mkstemp(prefix="curl-headers-", suffix=".tmp")
  os.close(header_fd)
  header_path = Path(header_name)
  if output is None:
    body_fd, body_name = tempfile.mkstemp(prefix="curl-body-", suffix=".tmp")
    os.close(body_fd)
    body_path = Path(body_name)
  else:
    body_path = Path(output)
  config_path = curl_config(headers or {}, credentials) if headers or credentials else None
  args = [
    "curl",
    "--silent",
    "--show-error",
    "--connect-timeout",
    "20",
    "--speed-limit",
    "1024",
    "--speed-time",
    "30",
    "--dump-header",
    str(header_path),
    "--output",
    str(body_path),
    "--write-out",
    "%{http_code}",
  ]
  if retry:
    args += ["--retry", "3", "--retry-all-errors", "--retry-delay", "1"]
  if config_path:
    args += ["--config", str(config_path)]
  else:
    args.append("--location")
  args.append("--head" if method == "HEAD" else "--request")
  if method != "HEAD":
    args.append(method)
  if byte_range:
    args += ["--range", byte_range]
  if data is not None:
    args += ["--data-binary", "@-"]
  elif upload is not None:
    args += ["--upload-file", str(upload)]
  args.append(url)

  try:
    result = subprocess.run(args, input=data, capture_output=True)
    status = int(result.stdout[-3:] or b"0") if result.returncode == 0 else 0
    response_headers = parse_headers(header_path)
    body = body_path.read_bytes() if output is None else b""
    return status, response_headers, body
  finally:
    header_path.unlink(missing_ok=True)
    if output is None:
      body_path.unlink(missing_ok=True)
    if config_path:
      config_path.unlink(missing_ok=True)


def public_url(oid):
  return f"{PUBLIC_BASE}/sha256/{oid}"


def public_object_exists(oid, size):
  for attempt in range(RETRIES):
    status, headers, _body = curl("HEAD", public_url(oid))
    if status == 200:
      try:
        actual_size = int(headers["content-length"])
      except (KeyError, ValueError) as error:
        raise RuntimeError(f"missing Content-Length for {oid}") from error
      if actual_size != size:
        raise RuntimeError(f"wrong remote size for {oid}: expected {size}, got {actual_size}")
      return True
    if status == 404:
      return False
    if status not in (0, 429, 500, 502, 503, 504):
      raise RuntimeError(f"public HEAD returned HTTP {status} for {oid}")
    time.sleep(min(2**attempt, 8))
  raise RuntimeError(f"public HEAD failed for {oid}")


def download_one(oid, size):
  dst = object_path(oid)
  with Lock(lock_path(oid)):
    if valid(dst, oid, size):
      return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{oid}-{os.getpid()}.tmp")
    href = public_url(oid)
    try:
      ranges = min(RANGES, max(1, (size + RANGE_MIN - 1) // RANGE_MIN))
      if ranges > 1:
        spans = [(i * size // ranges, (i + 1) * size // ranges - 1) for i in range(ranges)]
        parts = [dst.with_name(f"{oid}-{os.getpid()}-{i}.tmp") for i in range(ranges)]

        def fetch(item):
          part, span = item
          return curl("GET", href, output=part, byte_range=f"{span[0]}-{span[1]}", retry=True)[0]

        with concurrent.futures.ThreadPoolExecutor(max_workers=ranges) as pool:
          statuses = list(pool.map(fetch, zip(parts, spans, strict=True)))
        if all(
          status == 206 and part.is_file() and part.stat().st_size == end - start + 1 for status, part, (start, end) in zip(statuses, parts, spans, strict=True)
        ):
          with tmp.open("wb") as out:
            for part in parts:
              with part.open("rb") as stream:
                shutil.copyfileobj(stream, out)
        else:
          for part in parts:
            part.unlink(missing_ok=True)
          status, _headers, _body = curl("GET", href, output=tmp, retry=True)
          if not 200 <= status < 300:
            raise RuntimeError(f"download returned HTTP {status}")
        for part in parts:
          part.unlink(missing_ok=True)
      else:
        status, _headers, _body = curl("GET", href, output=tmp, retry=True)
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
  with concurrent.futures.ThreadPoolExecutor(max_workers=TRANSFERS) as pool:
    for future in [pool.submit(download_one, oid, size) for oid, size in missing]:
      future.result()


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
  assert proc.stdin is not None and proc.stdout is not None
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


class ObjectExists(Exception):
  pass


class R2Client:
  def credentials(self):
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
      raise RuntimeError("R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY are required for uploads")
    return access_key, secret_key

  def request(self, method, oid, *, query=(), payload_hash=EMPTY_SHA256, data=None, upload=None, content_length=None):
    path = f"/{BUCKET}/sha256/{oid}"
    encoded_query = urllib.parse.urlencode(sorted(query))
    url = S3_ENDPOINT + path
    if encoded_query:
      url += "?" + encoded_query
    headers = {"x-amz-content-sha256": payload_hash}
    if content_length is not None:
      headers["Content-Length"] = str(content_length)
      headers["Expect"] = ""
    return curl(method, url, headers=headers, credentials=self.credentials(), data=data, upload=upload)


def retryable(status):
  return status in (0, 429, 500, 502, 503, 504)


def wait_for_public_object(oid, size, attempts=RETRIES):
  for attempt in range(attempts):
    if public_object_exists(oid, size):
      return True
    time.sleep(min(2**attempt, 8))
  return False


def xml_value(body, name):
  try:
    element = ET.fromstring(body).find(f".//{{*}}{name}")
  except ET.ParseError as error:
    raise RuntimeError("R2 returned malformed XML") from error
  if element is None or element.text is None:
    raise RuntimeError(f"R2 response has no {name}")
  return element.text


def prepare_part(path, offset, size, part_number):
  directory = GIT_DIR / "lfs" / "tmp"
  directory.mkdir(parents=True, exist_ok=True)
  fd, name = tempfile.mkstemp(prefix=f"r2-part-{part_number}-", suffix=".tmp", dir=directory)
  part_path = Path(name)
  sha256 = hashlib.sha256()
  md5 = hashlib.md5(usedforsecurity=False)
  try:
    with path.open("rb") as source, os.fdopen(fd, "wb") as part:
      source.seek(offset)
      remaining = size
      while remaining:
        chunk = source.read(min(1 << 20, remaining))
        if not chunk:
          raise RuntimeError(f"short read from {path}")
        part.write(chunk)
        sha256.update(chunk)
        md5.update(chunk)
        remaining -= len(chunk)
    return part_path, sha256.hexdigest(), md5.hexdigest()
  except BaseException:
    part_path.unlink(missing_ok=True)
    raise


def create_multipart_upload(client, oid, size):
  for attempt in range(RETRIES):
    status, _headers, body = client.request("POST", oid, query=(("uploads", ""),), data=b"", content_length=0)
    if status == 200:
      return xml_value(body, "UploadId")
    if not retryable(status):
      raise RuntimeError(f"CreateMultipartUpload returned HTTP {status} for {oid}")
    if public_object_exists(oid, size):
      raise ObjectExists
    time.sleep(min(2**attempt, 8))
  raise RuntimeError(f"CreateMultipartUpload failed for {oid}")


def upload_part(client, path, oid, size, upload_id, part_number, offset, part_size):
  part_path, part_sha256, part_md5 = prepare_part(path, offset, part_size, part_number)
  query = (("partNumber", part_number), ("uploadId", upload_id))
  try:
    for attempt in range(RETRIES):
      status, headers, _body = client.request(
        "PUT",
        oid,
        query=query,
        payload_hash=part_sha256,
        upload=part_path,
        content_length=part_size,
      )
      if status == 200:
        etag = headers.get("etag", "").strip('"').lower()
        if etag != part_md5:
          raise RuntimeError(f"wrong ETag for part {part_number} of {oid}")
        return part_number, etag
      if not retryable(status):
        raise RuntimeError(f"UploadPart {part_number} returned HTTP {status} for {oid}")
      if public_object_exists(oid, size):
        raise ObjectExists
      time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"UploadPart {part_number} failed for {oid}")
  finally:
    part_path.unlink(missing_ok=True)


def abort_multipart_upload(client, oid, upload_id):
  for attempt in range(RETRIES):
    status, _headers, _body = client.request("DELETE", oid, query=(("uploadId", upload_id),))
    if status in (204, 404):
      return
    if not retryable(status):
      raise RuntimeError(f"AbortMultipartUpload returned HTTP {status} for {oid}")
    time.sleep(min(2**attempt, 8))
  raise RuntimeError(f"AbortMultipartUpload failed for {oid}")


def complete_multipart_upload(client, oid, size, upload_id, parts):
  root = ET.Element("CompleteMultipartUpload")
  for part_number, etag in sorted(parts):
    part = ET.SubElement(root, "Part")
    ET.SubElement(part, "PartNumber").text = str(part_number)
    ET.SubElement(part, "ETag").text = f'"{etag}"'
  body = ET.tostring(root, encoding="utf-8", xml_declaration=True)
  expected_etag = hashlib.md5(b"".join(bytes.fromhex(etag) for _number, etag in sorted(parts)), usedforsecurity=False).hexdigest() + f"-{len(parts)}"
  query = (("uploadId", upload_id),)
  for attempt in range(RETRIES):
    status, _headers, response_body = client.request(
      "POST",
      oid,
      query=query,
      payload_hash=hashlib.sha256(body).hexdigest(),
      data=body,
      content_length=len(body),
    )
    if status == 200:
      etag = xml_value(response_body, "ETag").strip('"').lower()
      if etag != expected_etag:
        raise RuntimeError(f"wrong multipart ETag for {oid}")
      if not wait_for_public_object(oid, size):
        raise RuntimeError(f"completed object {oid} is not publicly readable")
      return
    if not retryable(status):
      raise RuntimeError(f"CompleteMultipartUpload returned HTTP {status} for {oid}")
    if wait_for_public_object(oid, size, attempts=2):
      raise ObjectExists
    time.sleep(min(2**attempt, 8))
  raise RuntimeError(f"CompleteMultipartUpload failed for {oid}")


def multipart_upload(client, path, oid, size):
  upload_id = create_multipart_upload(client, oid, size)
  parts = [(part_number + 1, offset, min(PART_SIZE, size - offset)) for part_number, offset in enumerate(range(0, size, PART_SIZE))]
  try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=UPLOAD_CONCURRENCY) as pool:
      futures = [pool.submit(upload_part, client, path, oid, size, upload_id, part_number, offset, part_size) for part_number, offset, part_size in parts]
      uploaded_parts = [future.result() for future in futures]
    complete_multipart_upload(client, oid, size, upload_id, uploaded_parts)
  except BaseException:
    abort_multipart_upload(client, oid, upload_id)
    raise


def single_upload(client, path, oid, size):
  _sha256, expected_etag = hash_region(path, 0, size)
  for attempt in range(RETRIES):
    status, headers, _body = client.request("PUT", oid, payload_hash=oid, upload=path, content_length=size)
    if status == 200:
      etag = headers.get("etag", "").strip('"').lower()
      if etag != expected_etag:
        raise RuntimeError(f"wrong ETag for {oid}")
      if not wait_for_public_object(oid, size):
        raise RuntimeError(f"uploaded object {oid} is not publicly readable")
      return
    if not retryable(status):
      raise RuntimeError(f"PutObject returned HTTP {status} for {oid}")
    if wait_for_public_object(oid, size, attempts=2):
      return
    time.sleep(min(2**attempt, 8))
  raise RuntimeError(f"PutObject failed for {oid}")


def upload_one(client, oid, size):
  path = object_path(oid)
  if not valid(path, oid, size):
    raise RuntimeError(f"missing local LFS object {oid}")
  if hash_file(path) != oid:
    raise RuntimeError(f"local sha256 mismatch for {oid}")
  if public_object_exists(oid, size):
    return False
  try:
    if size < MULTIPART_THRESHOLD:
      single_upload(client, path, oid, size)
    else:
      multipart_upload(client, path, oid, size)
  except ObjectExists:
    if not public_object_exists(oid, size):
      raise
  return True


def upload_many(objects):
  objects = list(dict.fromkeys(objects))
  if not objects:
    return []
  client = R2Client()
  with concurrent.futures.ThreadPoolExecutor(max_workers=TRANSFERS) as pool:
    futures = [pool.submit(upload_one, client, oid, size) for oid, size in objects]
    return [future.result() for future in futures]


def push(ref="HEAD"):
  upload_many(pointers_for_ref(ref).keys())


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
  upload_many(objects)


def main():
  command = sys.argv[1] if len(sys.argv) > 1 else ""
  if command == "filter-process":
    filter_process()
  elif command == "pull":
    args = sys.argv[2:]
    exclude = next((arg.split("=", 1)[1] for arg in args if arg.startswith("--exclude=")), None)
    pull(next((arg for arg in args if not arg.startswith("--")), "HEAD"), exclude)
  elif command == "push":
    push(sys.argv[2] if len(sys.argv) > 2 else "HEAD")
  elif command == "pre-push":
    pre_push(sys.argv[2] if len(sys.argv) > 2 else "origin")
  elif command == "install":
    install()
  else:
    raise SystemExit("usage: lfs.py {filter-process|pull|push|pre-push|install}")


if __name__ == "__main__":
  main()
