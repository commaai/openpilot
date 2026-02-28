"""Local content-addressed LFS object cache."""

import hashlib
import os
import tempfile

from openpilot.tools.uv_lfs.config import git_dir


def objects_dir() -> str:
  return os.path.join(git_dir(), "uv-lfs", "objects")


def object_path(oid: str) -> str:
  return os.path.join(objects_dir(), oid[:2], oid[2:4], oid)


def has_object(oid: str) -> bool:
  return os.path.isfile(object_path(oid))


def read_object(oid: str) -> bytes:
  with open(object_path(oid), "rb") as f:
    return f.read()


def store_object(oid: str, data: bytes) -> None:
  """Store data in the local cache, verifying sha256."""
  actual = hashlib.sha256(data).hexdigest()
  if actual != oid:
    raise ValueError(f"sha256 mismatch: expected {oid}, got {actual}")

  path = object_path(oid)
  os.makedirs(os.path.dirname(path), exist_ok=True)

  # atomic write
  fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
  closed = False
  try:
    os.write(fd, data)
    os.close(fd)
    closed = True
    os.replace(tmp, path)
  except BaseException:
    if not closed:
      os.close(fd)
    try:
      os.unlink(tmp)
    except OSError:
      pass
    raise


def remove_object(oid: str) -> bool:
  """Remove an object from the cache. Returns True if it existed."""
  path = object_path(oid)
  try:
    os.unlink(path)
    return True
  except FileNotFoundError:
    return False


def cached_oids() -> set[str]:
  """Return the set of all cached OIDs."""
  base = objects_dir()
  oids: set[str] = set()
  if not os.path.isdir(base):
    return oids
  for d1 in os.listdir(base):
    p1 = os.path.join(base, d1)
    if not os.path.isdir(p1):
      continue
    for d2 in os.listdir(p1):
      p2 = os.path.join(p1, d2)
      if not os.path.isdir(p2):
        continue
      for name in os.listdir(p2):
        if len(name) == 64:
          oids.add(name)
  return oids
