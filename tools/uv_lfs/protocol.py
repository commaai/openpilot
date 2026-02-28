"""Git long-running filter-process protocol (pkt-line format)."""

import hashlib
import sys

from openpilot.tools.uv_lfs.batch_api import download_objects
from openpilot.tools.uv_lfs.pointer import format_pointer, parse_pointer
from openpilot.tools.uv_lfs.storage import has_object, read_object, store_object

# pkt-line constants
FLUSH = b"0000"
MAX_PKT_DATA = 65516


def pkt_line(data: bytes) -> bytes:
  """Encode a pkt-line."""
  length = len(data) + 4
  return f"{length:04x}".encode() + data


def pkt_line_text(text: str) -> bytes:
  """Encode a text pkt-line (with newline)."""
  return pkt_line((text + "\n").encode())


def _read_exact(stream, n: int) -> bytes:
  """Read exactly n bytes from stream, raising EOFError if short."""
  data = stream.read(n)
  if len(data) < n:
    raise EOFError("unexpected end of pkt-line stream")
  return data


def read_pkt_line(stream) -> bytes | None:
  """Read a single pkt-line. Returns None for flush packet."""
  header = stream.read(4)
  if not header:
    raise EOFError("unexpected end of pkt-line stream")
  if len(header) < 4:
    raise EOFError("truncated pkt-line header")
  if header == b"0000":
    return None
  try:
    length = int(header, 16)
  except ValueError as e:
    raise EOFError(f"invalid pkt-line header: {header!r}") from e
  if length <= 4:
    return b""
  return _read_exact(stream, length - 4)


def read_pkt_lines_until_flush(stream) -> list[bytes]:
  """Read pkt-lines until a flush packet."""
  lines = []
  while True:
    line = read_pkt_line(stream)
    if line is None:
      return lines
    lines.append(line)


def read_content_until_flush(stream) -> bytes:
  """Read binary content packets until flush."""
  chunks = []
  while True:
    line = read_pkt_line(stream)
    if line is None:
      return b"".join(chunks)
    chunks.append(line)


def write_flush(stream) -> None:
  stream.write(b"0000")


def write_status(stream, status: str) -> None:
  """Write a status response."""
  stream.write(pkt_line_text(f"status={status}"))
  write_flush(stream)


def write_content(stream, data: bytes) -> None:
  """Write content in pkt-line chunks followed by flush."""
  offset = 0
  while offset < len(data):
    chunk = data[offset:offset + MAX_PKT_DATA]
    stream.write(pkt_line(chunk))
    offset += len(chunk)
  write_flush(stream)


def run_filter_process() -> None:
  """Main loop for `git filter-process` long-running protocol."""
  stdin = sys.stdin.buffer
  stdout = sys.stdout.buffer

  # handshake: git sends "git-filter-client\n", version=2, flush
  lines = read_pkt_lines_until_flush(stdin)
  if not lines or lines[0].strip() != b"git-filter-client":
    raise RuntimeError("expected git-filter-client handshake")

  # respond with server handshake
  stdout.write(pkt_line_text("git-filter-server"))
  stdout.write(pkt_line_text("version=2"))
  write_flush(stdout)
  stdout.flush()

  # capability exchange: git sends capabilities, flush
  caps = read_pkt_lines_until_flush(stdin)
  cap_set = {c.strip().decode() for c in caps}

  # respond with our capabilities
  if "capability=clean" in cap_set:
    stdout.write(pkt_line_text("capability=clean"))
  if "capability=smudge" in cap_set:
    stdout.write(pkt_line_text("capability=smudge"))
  write_flush(stdout)
  stdout.flush()

  # command loop
  while True:
    try:
      meta_lines = read_pkt_lines_until_flush(stdin)
    except EOFError:
      break
    if not meta_lines:
      break

    meta = {}
    for line in meta_lines:
      text = line.decode().strip()
      if "=" in text:
        k, v = text.split("=", 1)
        meta[k] = v

    command = meta.get("command", "")
    content = read_content_until_flush(stdin)

    if command == "clean":
      result = _do_clean(content)
    elif command == "smudge":
      result = _do_smudge(content)
    else:
      write_status(stdout, "error")
      stdout.flush()
      continue

    if result is not None:
      write_status(stdout, "success")
      write_content(stdout, result)
      write_flush(stdout)  # empty second list = keep status unchanged
    else:
      write_status(stdout, "error")
    stdout.flush()


def _do_clean(content: bytes) -> bytes:
  """Clean filter: working tree → git storage. Returns pointer if large file."""
  # if it's already a pointer, pass through
  parsed = parse_pointer(content)
  if parsed is not None:
    return content

  # compute sha256 and create pointer
  oid = hashlib.sha256(content).hexdigest()
  size = len(content)

  # cache the real content
  if not has_object(oid):
    store_object(oid, content)

  return format_pointer(oid, size)


def _do_smudge(content: bytes) -> bytes | None:
  """Smudge filter: git storage → working tree. Returns real content or None on error."""
  parsed = parse_pointer(content)
  if parsed is None:
    # not a pointer, pass through
    return content

  oid, size = parsed

  # check local cache
  if has_object(oid):
    return read_object(oid)

  # try to download
  try:
    downloaded = download_objects([(oid, size)], progress=False)
    if downloaded > 0 and has_object(oid):
      return read_object(oid)
  except Exception:
    pass

  # return pointer as-is if download fails (don't block checkout)
  return content
