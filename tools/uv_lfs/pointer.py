"""Parse and generate LFS pointer files."""

LFS_POINTER_VERSION = "https://git-lfs.github.com/spec/v1"
POINTER_MAX_SIZE = 200  # pointers are always small


def parse_pointer(data: bytes) -> tuple[str, int] | None:
  """Parse an LFS pointer, returning (oid, size) or None if not a pointer."""
  if len(data) > POINTER_MAX_SIZE:
    return None
  try:
    text = data.decode("utf-8")
  except UnicodeDecodeError:
    return None

  lines = text.strip().split("\n")
  if len(lines) < 3:
    return None
  if lines[0] != f"version {LFS_POINTER_VERSION}":
    return None

  oid = None
  size = None
  for line in lines[1:]:
    if line.startswith("oid sha256:"):
      oid = line[len("oid sha256:"):]
    elif line.startswith("size "):
      try:
        size = int(line[5:])
      except ValueError:
        return None

  if oid is None or size is None:
    return None
  if len(oid) != 64:
    return None
  return oid, size


def format_pointer(oid: str, size: int) -> bytes:
  """Generate an LFS pointer file."""
  return f"version {LFS_POINTER_VERSION}\noid sha256:{oid}\nsize {size}\n".encode()
