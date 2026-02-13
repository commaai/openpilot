#!/usr/bin/env python3
import os
import sys
import glob
import hashlib
import pickle
from pathlib import Path

def make_external_pickle(raw_path: str, out_prefix: str, chunk_bytes: int) -> None:
  print(f"splitting {raw_path} into {chunk_bytes} byte chunks with prefix {out_prefix}")
  data = Path(raw_path).read_bytes()

  out_dir = os.path.dirname(out_prefix) or "."
  base = os.path.basename(out_prefix)

  keep = set()
  lines = []
  for i in range(0, len(data), chunk_bytes):
    chunk = data[i:i + chunk_bytes]
    name = f"{base}.data-{(i // chunk_bytes) + 1:03d}"
    path = os.path.join(out_dir, name)
    Path(path).write_bytes(chunk)
    keep.add(path)
    lines.append(f"{name}\t{hashlib.sha256(chunk).hexdigest()}")

  # delete stale chunks from older chunk sizes
  for p in glob.glob(out_prefix + ".data-*"):
    if p not in keep:
      try:
        os.unlink(p)
      except FileNotFoundError:
        pass

  parts_path = out_prefix + ".parts"
  Path(parts_path).write_text(
    hashlib.sha256(data).hexdigest() + "\n" +
    str(len(data)) + "\n" +
    "\n".join(lines) + ("\n" if lines else ""),
    encoding="utf-8",
  )


def load_external_pickle(prefix_or_parts: str) -> bytes:
  print(f"loading external pickle from {prefix_or_parts}")
  parts_path = prefix_or_parts if prefix_or_parts.endswith(".parts") else (prefix_or_parts + ".parts")
  base_dir = os.path.dirname(parts_path) or "."

  lines = Path(parts_path).read_text(encoding="utf-8").splitlines()
  if len(lines) < 2:
    raise RuntimeError("bad manifest (need at least 2 lines)")

  full_expected = lines[0].strip()
  out = bytearray()
  for ln in lines[2:]:
    if not ln.strip():
      continue
    name, _ = ln.split("\t", 1)
    out += Path(os.path.join(base_dir, name)).read_bytes()

  if hashlib.sha256(bytes(out)).hexdigest() != full_expected:
    raise RuntimeError("full hash mismatch")
  return pickle.loads(bytes(out))

if __name__ == "__main__":
  make_external_pickle(sys.argv[1], sys.argv[2], int(sys.argv[3]))
